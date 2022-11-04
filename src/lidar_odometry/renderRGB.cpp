#include "renderRGB.hpp"
#include "utility.h"

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
pcl::PointCloud<PointType>::Ptr framePoses3D;            // 当前图片帧的位置
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
    //image相关
    ros::Subscriber subKeyFramePoses;
    RGB() {
        subKeyFramePoses = nh.subscribe<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 5, &RGB::keyFramePosesHandler, this, ros::TransportHints().tcpNoDelay());
        // subKeyFramePoses    = nh.subscribe<sensor_msgs::PointCloud2>     (PROJECT_NAME + "/lidar/mapping/trajectory", 5, &RGB::keyFramePosesHandler, this, ros::TransportHints().tcpNoDelay());
    }
    /**
     * @brief 关键帧位姿回调函数，
     * 
     */
    void keyFramePosesHandler(const nav_msgs::Odometry::ConstPtr& msgIn) {//注意，回调函数要加ConstPtr
        //提取当前帧时间戳
        timeCurKFStamp = msgIn->header.stamp;
        // timeCurKF = msgIn->header.stamp.toSec();
        timeCurKF = ROS_TIME(msgIn);//取出时间戳
        printf("current KeyFrame timestamp: %f.\n", timeCurKF);

    }
};

// void rednerRGB() {
//     // 附近的keyframes (最后一个keyframe附近, 50m)
//     pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
//     pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
//     std::vector<int> pointSearchInd;     // keyframes的index
//     std::vector<float> pointSearchSqDis; // keyframes的距离

//     // 1.extract all the nearby key poses and downsample them
//     kdtreeSurroundingKeyPoses->setInputCloud(framePoses3D); // create kd-tree
//     //根据最后一个关键帧位置，在一定距离内搜索关键帧
//     kdtreeSurroundingKeyPoses->radiusSearch(framePoses3D->back(), (double)50.0, pointSearchInd, pointSearchSqDis);
//     //将附近关键帧点云存入surroundingKeyPoses中
//     for (int i = 0; i < (int)pointSearchInd.size(); ++i)
//     {
//         int id = pointSearchInd[i];
//         surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
//     }
//     //避免关键帧过多，做一个下采样
//     downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
//     downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

//     // 2.also extract some latest key frames in case the robot rotates in one position
//     //也提取时间上较近的关键帧
//     int numPoses = cloudKeyPoses3D->size();
//     for (int i = numPoses-1; i >= 0; --i)
//     {
//         if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0) // 10s内的keyframes
//             surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
//         else
//             break;
//     }

//     // 通过提取到的keyframes, 来提取点云, 从而构造localmap
//     extractCloud(surroundingKeyPosesDS);
// }



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