#include "renderRGB.hpp"
#include "utility.h"
#include "lvi_sam/cloud_info.h"

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;
//
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses; // 用来寻找附近的关键帧 (scan to map)
// pcl::PointCloud<PointType>::Ptr framePoses3D;               // 当前图片帧的位置
/**
 * Step 1：订阅关键帧位姿和特征点
 * 
 */
class RGB  : public ParamServer {
public:
    //lidar相关
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;            // 关键帧的位置 (intensity为keyframe的index)
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;        // 关键帧的位姿
    ros::Time timeCurKFStamp;                                   // 当前帧时间戳   
    double timeCurKF;                                           // 当前帧的时间戳 (double格式起始时刻)                
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;// for surrounding key poses of scan-to-map optimization
    
    ros::Subscriber subLaserCloudInfo;// 订阅关键帧Info

    ros::Publisher pubLocalPointCloud;// 发布 localmap的surface点云

    // 所有keyframs的点云
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 关键帧的corner特征 (这里是世界系, 和mapOptmization不同)
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 关键帧的surface特征
    
    // localmap的特征点云(map系), 用来进行scan to map的匹配
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;
    pcl::PointCloud<PointType>::Ptr localPointCloud;//局部地图点
    pcl::VoxelGrid<PointType> downSizeFilterCorner;//降采样边缘点
    pcl::VoxelGrid<PointType> downSizeFilterSurf;//降采样平面点
    //image相关
    
    
    //构造函数
    RGB() {
        // subKeyFramePoses = nh.subscribe<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 5, &RGB::keyFramePosesHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/mapping/KeyFrameInfo", 5, &RGB::keyFrmaeInfoHandler, this, ros::TransportHints().tcpNoDelay());
        
        pubLocalPointCloud = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/localPointCloud", 1);            // localmap的特征点云
        /// 分配内存
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());
        localPointCloud.reset(new pcl::PointCloud<PointType>());
        /// 初始化降采样尺寸 
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization 2m
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    }
    /**
     * @brief 关键帧相关信息回调函数
     * 
     */
    void keyFrmaeInfoHandler(const lvi_sam::cloud_infoConstPtr& KF_Info) {
        // Step 1：提取时间戳，位置姿态，特征点云
        static int iKeyFrame_ID = -1;
        iKeyFrame_ID++;//关键帧ID加一
        std::cout << "🚀⭐⭐⭐🌟🌟🌟✨✨✨✨✨✨🌟🌟🌟⭐⭐⭐🔭" << std::endl;
        std::cout << "当前关键帧ID为:" << iKeyFrame_ID << std::endl;
        timeCurKFStamp = KF_Info->header.stamp;
        // timeCurKF = msgIn->header.stamp.toSec();
        timeCurKF = ROS_TIME(KF_Info);//取出时间戳
        printf("时间戳为: %f.\n", timeCurKF);
        
        // 更新关键帧的位姿
        PointType thisPose3D;
        PointTypePose thisPose6D;
        
        thisPose3D.x = KF_Info->odomX;
        thisPose3D.y = KF_Info->odomY;
        thisPose3D.z = KF_Info->odomZ;
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index, intensity为关键帧的index
        cloudKeyPoses3D->push_back(thisPose3D); // 关键帧的位置
        
        // printf("当前关键帧yaw角为: %f.\n", yaw);
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = KF_Info->odomRoll;
        thisPose6D.pitch = KF_Info->odomPitch;
        thisPose6D.yaw   = KF_Info->odomYaw;
        thisPose6D.time = timeCurKF;
        cloudKeyPoses6D->push_back(thisPose6D); // 关键帧的位姿

        //特征点云
        pcl::PointCloud<PointType>::Ptr tempCloud;
        tempCloud.reset(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(KF_Info->cloud_corner, *tempCloud);
        cornerCloudKeyFrames.push_back(tempCloud);
        tempCloud.reset(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(KF_Info->cloud_surface, *tempCloud);
        surfCloudKeyFrames.push_back(tempCloud);
        

        
        // Step 2：提取附近的keyframes及其点云, 来构造localmap
        // 附近的keyframes (最后一个keyframe附近, 50m)
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;     // keyframes的index
        std::vector<float> pointSearchSqDis; // keyframes的距离

        // 2.1 extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        //根据最后一个关键帧位置，在一定距离内搜索关键帧 50m
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)10.0, pointSearchInd, pointSearchSqDis);//surroundingKeyframeSearchRadius
        //将附近关键帧点云存入surroundingKeyPoses中
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }
        //避免关键帧过多，做一个下采样 间距为2m
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // 2.2 also extract some latest key frames in case the robot rotates in one position
        // 也提取时间上较近的关键帧
        // int numPoses = cloudKeyPoses3D->size();
        // for (int i = numPoses - 1; i >= 0; --i)
        // {
        //     if (timeCurKF - cloudKeyPoses6D->points[i].time < 10.0) // 10s内的keyframes
        //         surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
        //     else
        //         break;
        // }
        std::cout << "找到的局部关键帧个数为：" << surroundingKeyPosesDS->size() << std::endl;
        ///提取附近点云，结果保存在localPointCloud中
        extractCloud(surroundingKeyPosesDS);
    }
    /**
     * @brief 通过提取到的keyframes, 来提取点云, 从而构造localmap
     * 
     */
     
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // 用于并行计算, 为每个keyframe提取点云
        std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

        laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

        // extract surrounding map
        // 1.并行计算, 分别提取每个keyframe的点云
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            int thisKeyInd = (int)cloudToExtract->points[i].intensity; // intensity为keyframe的index
            if (pointDistance(cloudKeyPoses3D->points[thisKeyInd], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            laserCloudCornerSurroundingVec[i]  = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            laserCloudSurfSurroundingVec[i]    = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
            // laserCloudCornerSurroundingVec[i] = *cornerCloudKeyFrames[thisKeyInd];
            // laserCloudSurfSurroundingVec[i] = *surfCloudKeyFrames[thisKeyInd];
        }///至此局部特征点云存储进两个vector

        // 2.fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            *laserCloudSurfFromMap   += laserCloudSurfSurroundingVec[i];
        }

        // 3.分别对Corner和Surface特征进行采样
        
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);

        // 4.提取局部地图点，并发布出去
        localPointCloud->clear();
        *localPointCloud += *laserCloudCornerFromMapDS;
        *localPointCloud += *laserCloudSurfFromMapDS;
        std::cout << "局部地图点数量为：" << localPointCloud->size() << std::endl;
        publishCloud(&pubLocalPointCloud, localPointCloud, timeCurKFStamp, "odom");
    }
    // 点云坐标变换
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }
};





int main(int argc, char** argv)
{
    ros::init(argc, argv, "RGB");

    RGB rgb;

    ROS_INFO("\033[1;32m----> Lidar Map Optimization Started.\033[0m");
    
    // std::thread loopDetectionthread(&mapOptimization::loopClosureThread, &MO);
    // std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    // loopDetectionthread.join();
    // visualizeMapThread.join();

    return 0;
}