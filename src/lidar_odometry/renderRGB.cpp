#include "renderRGB.hpp"
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses; // 用来寻找附近的关键帧 (scan to map)
// pcl::PointCloud<PointType>::Ptr framePoses3D;               // 当前图片帧的位置
/**
 * Step 1：订阅关键帧位姿和特征点
 * 
 */
class RGB  : public ParamServer {
public:
    ///lidar相关
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;            // 关键帧的位置 (intensity为keyframe的index)
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;        // 关键帧的位姿
    ros::Time timeCurKFStamp;                                   // 当前帧时间戳   
    double timeCurKF;                                           // 当前帧的时间戳 (double格式起始时刻)                
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;// for surrounding key poses of scan-to-map optimization
    
    ros::Subscriber subLaserCloudInfo;  // 订阅关键帧Info

    ros::Publisher pubLocalPointCloud;  // 发布 localmap的surface点云
    ros::Publisher pubCloudInImage;     // 发布 处在相机视野范围内的lidar点
    ros::Publisher pubRGB_Cloud;        // 发布 RGB渲染之后的点云
    

    // 所有keyframs的点云
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;   // body系下关键帧的corner特征  ///TODO:不确定body系还是lidar系，二者可能略有区别
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;     // body系下关键帧的surface特征
    vector<pcl::PointCloud<PointType>::Ptr> deskewedCloudKeyFrames; // 去畸变的所有点云

    // localmap的特征点云(map系), 用来进行scan to map的匹配
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudDeskewedFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;
    pcl::PointCloud<PointType>::Ptr localPointCloud;    //局部地图点
    pcl::PointCloud<PointType>::Ptr cloudInImage;    //局部地图点
    pcl::VoxelGrid<PointType> downSizeFilterCorner;     //降采样边缘点
    pcl::VoxelGrid<PointType> downSizeFilterSurf;       //降采样平面点

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr RGB_Cloud;        //RGB渲染之后的点云
    ///image相关
    ros::Subscriber subImagePose;   //订阅图片位姿
    ros::Subscriber subImage;       //订阅原始图片
    double timeImage;
    queue<pair<cv::Mat, double>> images_buf;      //存放读取的原始图片
    std::mutex m_image;                 // 原始图片对应的锁
    std::mutex m_cloud;                 // 点云的锁
    double fx, fy, cx, cy;              // 相机内参
    int imgCols, imgRows;               // 图片尺寸


    
    //构造函数
    RGB() {
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/mapping/KeyFrameInfo", 5, &RGB::keyFrameInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subImagePose = nh.subscribe(PROJECT_NAME + "/vins/odometry/keyframe_pose",  3, &RGB::imagePoseCallback, this, ros::TransportHints().tcpNoDelay());
        subImage     = nh.subscribe("/camera/color/image_raw", 30, &RGB::imageCallBack, this, ros::TransportHints().tcpNoDelay());
        
        pubLocalPointCloud = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/localPointCloud", 1);            // localmap的特征点云
        pubCloudInImage = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloudInImage", 1);   
        pubRGB_Cloud = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/RGB_Cloud", 1);         // localmap的特征点云
        
        /// 分配内存
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudDeskewedFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());
        localPointCloud.reset(new pcl::PointCloud<PointType>());
        cloudInImage.reset(new pcl::PointCloud<PointType>());
        RGB_Cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        /// 初始化降采样尺寸 
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization 2m
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        // 初始化相机内参，根据yaml文件手动输入
        fx = 617.971050917033;
        fy = 616.445131524790;
        cx = 327.710279392468;
        cy = 253.976983707814;
        imgCols = 640;
        imgRows = 480;
    }
    
    
    /**
     * @brief 图片位姿回调函数
     *  1. 接收到一帧图片位姿时，进入主处理函数
        2. 在原始图片队列中，找到对应的原始图片
        3. 在历史点云关键帧中，找到附近的关键帧
        4. 将附近关键帧构成点云局部地图
        5. 将点云局部地图投影到图片上，进行颜色渲染
     */
    void imagePoseCallback(const nav_msgs::Odometry::ConstPtr &pose_msg)
    {
        static int keyImage_ID = -1;
        keyImage_ID++;//关键帧ID加一
        std::cout << "🚀⭐⭐⭐🌟🌟🌟✨✨✨✨✨✨🌟🌟🌟⭐⭐⭐🔭" << std::endl;
        std::cout << "当前图片关键帧ID为:" << keyImage_ID << std::endl;
        // Step 1：获得图片的时间戳、位置、姿态
        timeImage = ROS_TIME(pose_msg);
        printf("当前图片姿态时间戳为: %f.\n", timeImage);
        PointType imagePose3D;
        PointTypePose imagePose6D;
        imagePose6D.x = pose_msg->pose.pose.position.x;
        imagePose6D.y = pose_msg->pose.pose.position.y;
        imagePose6D.z = pose_msg->pose.pose.position.z;
        imagePose3D.x = imagePose6D.x;
        imagePose3D.y = imagePose6D.y;
        imagePose3D.z = imagePose6D.z;
        tf::Quaternion quat;
        tf::quaternionMsgToTF(pose_msg->pose.pose.orientation, quat);
        double roll, pitch, yaw;//定义存储r\p\y的容器
        tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);//进行转换
        imagePose6D.roll = roll;
        imagePose6D.pitch = pitch;
        imagePose6D.yaw = yaw;
        ///利用TF将图片位姿转换到世界坐标系
        // listen to transform 获取camera在世界坐标系的位姿(camera to world)        
        static tf::TransformListener listener;
        static tf::StampedTransform transform; // vins系到世界系
        try{
            listener.waitForTransform("odom", "vins_world", pose_msg->header.stamp, ros::Duration(0.01));
            listener.lookupTransform("odom", "vins_world", pose_msg->header.stamp, transform);
        } 
        catch (tf::TransformException ex){
            ROS_ERROR("lidar no tf");
        }

        double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
        xCur = transform.getOrigin().x();
        yCur = transform.getOrigin().y();
        zCur = transform.getOrigin().z();
        tf::Matrix3x3 m(transform.getRotation());
        m.getRPY(rollCur, pitchCur, yawCur);
        //vins_world系到Odom系变换
        Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);
        Eigen::Vector3f inPoseTemp = {imagePose3D.x, imagePose3D.y, imagePose3D.z};
        Eigen::Vector3f outPoseTemp;
        // PointType imagePose3DTrue;
        pcl::transformPoint(inPoseTemp, outPoseTemp, transNow);
        imagePose3D.x = outPoseTemp[0];
        imagePose3D.y = outPoseTemp[1];
        imagePose3D.z = outPoseTemp[2];

        // Step 2：根据时间戳，得到匹配的原始图片
        m_image.lock();
        pair<cv::Mat, double> curImg_Time;//找到的匹配的原始图片和时间戳
        while (!images_buf.empty()) {
            curImg_Time = images_buf.front();
            images_buf.pop();
                if (curImg_Time.second == timeImage) {
                // cout << "找到匹配的图片🤩🤩🤩🤩🤩🤩" << endl;
                    break;
            }
        }
        cv::Mat rawImage = curImg_Time.first;//原始图片
        m_image.unlock();
        // Step 3：在历史点云关键帧中，找到附近的关键帧
        m_cloud.lock();
        // 附近的keyframes (最后一个keyframe附近, 50m)
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;     // keyframes的index
        std::vector<float> pointSearchSqDis; // keyframes的距离
        /// 3.1 extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->radiusSearch(imagePose3D, (double)10.0, pointSearchInd, pointSearchSqDis);//surroundingKeyframeSearchRadius
        //将附近关键帧点云存入surroundingKeyPoses中
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }
        //避免关键帧过多，做一个下采样 间距为2m
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        /// 3.2 also extract some latest key frames in case the robot rotates in one position
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
        m_cloud.unlock();

        // Step 4：遍历点云，对相机视野内的点进行RGB渲染
        /// body系才是相机系
        // 获得从odom系到body系的位姿变换
        try{
            listener.waitForTransform("vins_body_ros", "odom", pose_msg->header.stamp, ros::Duration(0.01));
            listener.lookupTransform("vins_body_ros", "odom", pose_msg->header.stamp, transform);
        } 
        catch (tf::TransformException ex){
            ROS_ERROR("image no tf");
            // return depth_of_point;
        }

        // double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
        xCur = transform.getOrigin().x();
        yCur = transform.getOrigin().y();
        zCur = transform.getOrigin().z();
        tf::Matrix3x3 m1(transform.getRotation());
        m1.getRPY(rollCur, pitchCur, yawCur);
        //Eigen格式的Twb
        transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

        // transform cloud from global frame to camera frame
        pcl::PointCloud<PointType>::Ptr vinsLocalCloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*localPointCloud, *vinsLocalCloud, transNow);
        ///至此将body系的点云保存在了vinsLocalCloud中，后面要同时处理odom系和body系的点云
        int pointSize = vinsLocalCloud->size();
        cloudInImage->clear();
        RGB_Cloud->clear();
        //遍历局部地图中的所有点云
        for (int i = 0; i < pointSize; i++) {
            PointType tempCurPoint = (*vinsLocalCloud)[i];
            //做一下坐标轴的转换，从lidar的前左上转为image的右下前
            PointType curPoint;
            curPoint.x = -tempCurPoint.y;
            curPoint.y = -tempCurPoint.z;
            curPoint.z = tempCurPoint.x;
            if (curPoint.z < 0.01) continue;//跳过深度为负的点
            // cout << "执行到这里💫" << endl;
            double u, v;//投影的像素坐标
            u = fx * curPoint.x / curPoint.z + cx;
            v = fy * curPoint.y / curPoint.z + cy;
            //判断像素坐标是否落在图像内
            double scale = 0.01;//缩放系数，用于筛选小于原始图片大小的点
            if ((u < imgCols * scale + 1) || (u > imgCols * (1 - scale) - 1) ||
                (v < imgRows * scale + 1) || (v > imgRows * (1 - scale) - 1)) {
                    continue;//跳过不在图片范围内的点
                }
            cloudInImage->push_back((*localPointCloud)[i]);
            // cout << "该点像素坐标为\t" << u << "\t" << v << endl; 
            /// 获取该点的RGB，将结果保存进RGB_Cloud中
            int r = rawImage.at<cv::Vec3b>(v, u)[2];
            int g = rawImage.at<cv::Vec3b>(v, u)[1];
            int b = rawImage.at<cv::Vec3b>(v, u)[0];
            pcl::PointXYZRGB curPointRGB;
            curPointRGB.x = (*localPointCloud)[i].x;
            curPointRGB.y = (*localPointCloud)[i].y;
            curPointRGB.z = (*localPointCloud)[i].z;
            curPointRGB.r = r;
            curPointRGB.g = g;
            curPointRGB.b = b;
            RGB_Cloud->push_back(curPointRGB);

        }
        //发布处在相机视野范围内的lidar点
        publishCloud(&pubCloudInImage, cloudInImage, timeCurKFStamp, "odom");
        //发布RGB渲染的点云
        publishCloud(&pubRGB_Cloud, RGB_Cloud, timeCurKFStamp, "odom");

    }
    /**
     * @brief 原始图片回调函数，将原始图片和时间戳打包存入images_buf
     * 
     */
    void imageCallBack(const sensor_msgs::ImageConstPtr &img_msg) {
        m_image.lock();
        // 3.图像格式转换(ROS to OpenCV)
        cv_bridge::CvImageConstPtr ptr;
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::RGB8);

        cv::Mat show_img = ptr->image; // 得到原始图片
        
        images_buf.push({show_img, ROS_TIME(img_msg)}); //将图片放入队列中等待处理
        m_image.unlock();
    }


    /**
     * @brief 点云关键帧相关信息回调函数
     * 
     */
    void keyFrameInfoHandler(const lvi_sam::cloud_infoConstPtr& KF_Info) {
        // Step 1：提取时间戳，位置姿态，特征点云
        static int iKeyFrame_ID = -1;
        iKeyFrame_ID++;//关键帧ID加一
        // std::cout << "🚀⭐⭐⭐🌟🌟🌟✨✨✨✨✨✨🌟🌟🌟⭐⭐⭐🔭" << std::endl;
        // std::cout << "当前关键帧ID为:" << iKeyFrame_ID << std::endl;
        timeCurKFStamp = KF_Info->header.stamp;
        timeCurKF = ROS_TIME(KF_Info);//取出时间戳
        // printf("时间戳为: %f.\n", timeCurKF);
        
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

        tempCloud.reset(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(KF_Info->cloud_deskewed, *tempCloud);
        deskewedCloudKeyFrames.push_back(tempCloud);
        // cout << "一共有点云数量" << cloudKeyPoses3D->size() << endl;
        m_cloud.lock();
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree，为寻找邻近关键帧做准备
        m_cloud.unlock();
        
    }
    /**
     * @brief 通过提取到的keyframes, 来提取点云, 从而构造局部地图
     * ///最后结果保存在localPointCloud中
     */
     
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // 用于并行计算, 为每个keyframe提取点云
        std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudDeskewedSurroundingVec;
        laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        laserCloudSurfSurroundingVec.resize(cloudToExtract->size());
        laserCloudDeskewedSurroundingVec.resize(cloudToExtract->size());

        // extract surrounding map
        // 1.并行计算, 分别提取每个keyframe的点云
        // #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            int thisKeyInd = (int)cloudToExtract->points[i].intensity; // intensity为keyframe的index
            if (pointDistance(cloudKeyPoses3D->points[thisKeyInd], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            laserCloudCornerSurroundingVec[i]  = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            laserCloudSurfSurroundingVec[i]    = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
            laserCloudDeskewedSurroundingVec[i]= *transformPointCloud(deskewedCloudKeyFrames[thisKeyInd],&cloudKeyPoses6D->points[thisKeyInd]);
            
        }///至此局部特征点云存储进两个vector

        // 2.fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            *laserCloudSurfFromMap   += laserCloudSurfSurroundingVec[i];
            *laserCloudDeskewedFromMap += laserCloudDeskewedSurroundingVec[i];
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
        //LLH:选择一下是否使用下采样
        // *localPointCloud += *laserCloudCornerFromMapDS;
        // *localPointCloud += *laserCloudSurfFromMapDS;
        // *localPointCloud += *laserCloudCornerFromMap;
        // *localPointCloud += *laserCloudSurfFromMap;
        *localPointCloud += *laserCloudDeskewedFromMap;
        // std::cout << "局部地图点数量为：" << localPointCloud->size() << std::endl;
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