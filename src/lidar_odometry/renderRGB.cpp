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
    pcl::PointCloud<PointType>::Ptr cloudInImage;       //局部地图点
    pcl::VoxelGrid<PointType> downSizeFilterCorner;     //降采样边缘点
    pcl::VoxelGrid<PointType> downSizeFilterSurf;       //降采样平面点
    pcl::VoxelGrid<pcl::PointXYZRGB> downSizeFilterMapRGB;     //RGB点云地图降采样
    pcl::VoxelGrid<pcl::PointXYZRGB> downSizeFilterMapRGB2;     //RGB点云地图降采样2.0

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outRGBCloud;      //保存全部的RGB点云用于输出

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr RGB_Cloud;        //RGB渲染之后的点云
    ///image相关
    ros::Subscriber subImagePose;   //订阅图片位姿
    ros::Subscriber subImage;       //订阅原始图片
    double timeImage;
    queue<pair<cv::Mat, double>> images_buf;      //存放读取的原始图片
    std::mutex m_image;                 // 原始图片对应的锁
    std::mutex m_cloud;                 // 点云的锁
    double fx, fy, cx, cy;              // 相机内参
    double k1, k2, p1, p2;              // 畸变参数
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
        outRGBCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        /// 初始化降采样尺寸 
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization 2m
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterMapRGB.setLeafSize(frameVoxelSize, frameVoxelSize, frameVoxelSize);
        downSizeFilterMapRGB2.setLeafSize(RGBVoxelSize, RGBVoxelSize, RGBVoxelSize);
        // 读取相机参数
        ros::NodeHandle n;

        std::string config_file;
        n.getParam("vins_config_file", config_file);
        cv::FileStorage fs(config_file, cv::FileStorage::READ);

        imgCols = static_cast<int>(fs["image_width"]);
        imgRows = static_cast<int>(fs["image_height"]);

        cv::FileNode nn = fs["distortion_parameters"];
        k1 = static_cast<double>(nn["k1"]);
        k2 = static_cast<double>(nn["k2"]);
        p1 = static_cast<double>(nn["p1"]);
        p2 = static_cast<double>(nn["p2"]);

        nn = fs["projection_parameters"];
        fx = static_cast<double>(nn["fx"]);//llh：相机内参
        fy = static_cast<double>(nn["fy"]);
        cx = static_cast<double>(nn["cx"]);
        cy = static_cast<double>(nn["cy"]);
    
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
        // Step 1：获得图片的时间戳、位置、姿态，并将位置转换到世界系下
        timeImage = ROS_TIME(pose_msg);
        printf("当前图片姿态时间戳为: %f.\n", timeImage);
        PointType imagePose3D;      //相机位置
        PointTypePose imagePose6D;  //相机位置+姿态
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
        /// 在邻近范围搜寻关键帧基准点，并进行降采样
        kdtreeSurroundingKeyPoses->radiusSearch(imagePose3D, (double)20.0, pointSearchInd, pointSearchSqDis);//surroundingKeyframeSearchRadius
        //将附近关键帧点云存入surroundingKeyPoses中
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        // for (int i = 0; i < 1; ++i)// TODO 这里强行改成了只用最近一帧点云关键帧
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }
        // 避免关键帧过多，做一个下采样 间距为2m
        // downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        // downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        // std::cout << "找到的局部关键帧个数为：" << surroundingKeyPosesDS->size() << std::endl;
        ///提取附近点云，结果保存在localPointCloud中
        extractCloud(surroundingKeyPoses);
        m_cloud.unlock();

        // Step 4：遍历点云，对相机视野内的点进行RGB渲染
        // 获得从odom系到body系的位姿变换 body系才是相机系
        try{
            listener.waitForTransform("vins_camera", "odom", pose_msg->header.stamp, ros::Duration(0.01));
            listener.lookupTransform("vins_camera", "odom", pose_msg->header.stamp, transform);
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
        Eigen::Affine3f T_bo = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

        // transform cloud from global frame to camera frame
        pcl::PointCloud<PointType>::Ptr vinsLocalCloud(new pcl::PointCloud<PointType>());//存储body系下的局部地图点
        pcl::transformPointCloud(*localPointCloud, *vinsLocalCloud, T_bo);
        ///至此将body系的点云保存在了vinsLocalCloud中，后面要同时处理odom系和body系的点云
        int pointSize = vinsLocalCloud->size();
        cloudInImage->clear();
        RGB_Cloud->clear();
        //遍历局部地图中的所有点云
        for (int i = 0; i < (int)pointSize; i++) {
            PointType tempCurPoint = (*vinsLocalCloud)[i];
            //做一下坐标轴的转换，从lidar的前左上转为image的右下前
            PointType curPoint; //当前激光点
            curPoint.x = tempCurPoint.x;
            curPoint.y = tempCurPoint.y;
            curPoint.z = tempCurPoint.z;
            if (curPoint.z < 0.01 || curPoint.z > maxDistRGB) continue;//跳过深度为负的点，以及距离大于10m的点
            /// 从相机系点计算得到对应的像素坐标
            double u, v;//投影的像素坐标
            cameraProjective(curPoint, u, v);
            // u = fx * curPoint.x / curPoint.z + cx;
            // v = fy * curPoint.y / curPoint.z + cy;
            
            //判断像素坐标是否落在图像内
            double scale = 0.05;//缩放系数，用于筛选小于原始图片大小的点
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
        // Step 5：发布相关点云
        //发布处在相机视野范围内的lidar点
        // publishCloud(&pubCloudInImage, cloudInImage, timeCurKFStamp, "odom");
        //发布RGB渲染的点云
        publishCloud(&pubRGB_Cloud, RGB_Cloud, timeCurKFStamp, "odom");

        //最后保存此次全部RGB点云
        downSizeFilterMapRGB.setInputCloud(RGB_Cloud);
        downSizeFilterMapRGB.filter(*RGB_Cloud);
        for (int i = 0; i < (int)RGB_Cloud->size(); i++) {
            outRGBCloud->push_back(RGB_Cloud->at(i));
        }
        downSizeFilterMapRGB2.setInputCloud(outRGBCloud);
        downSizeFilterMapRGB2.filter(*outRGBCloud);
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
        m_cloud.lock();
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
        // tempCloud.reset(new pcl::PointCloud<PointType>());
        // pcl::fromROSMsg(KF_Info->cloud_corner, *tempCloud);
        // cornerCloudKeyFrames.push_back(tempCloud);

        // tempCloud.reset(new pcl::PointCloud<PointType>());
        // pcl::fromROSMsg(KF_Info->cloud_surface, *tempCloud);
        // surfCloudKeyFrames.push_back(tempCloud);

        
        tempCloud.reset(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(KF_Info->cloud_deskewed, *tempCloud);
        deskewedCloudKeyFrames.push_back(tempCloud);
        // cout << "一共有点云数量" << cloudKeyPoses3D->size() << endl;
        // Step 2：仅保留最近20帧的点云信息，送入kd树准备检索
        static int indexToDelete = 0;//即将清空的关键帧点云索引
        if (deskewedCloudKeyFrames.size() > historyCloudSize) {
            deskewedCloudKeyFrames[indexToDelete++].reset();
        }
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
        // std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        // std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudDeskewedSurroundingVec;
        // laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        // laserCloudSurfSurroundingVec.resize(cloudToExtract->size());
        laserCloudDeskewedSurroundingVec.resize(cloudToExtract->size());

        // extract surrounding map
        // 1.并行计算, 分别提取每个keyframe的点云
        // #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            int thisKeyInd = (int)cloudToExtract->points[i].intensity; // intensity为keyframe的index
            if (pointDistance(cloudKeyPoses3D->points[thisKeyInd], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            // laserCloudCornerSurroundingVec[i]  = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            // laserCloudSurfSurroundingVec[i]    = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
            if (deskewedCloudKeyFrames[thisKeyInd] != nullptr)
                laserCloudDeskewedSurroundingVec[i]= *transformPointCloud(deskewedCloudKeyFrames[thisKeyInd],&cloudKeyPoses6D->points[thisKeyInd]);
            
        }///至此局部特征点云存储进两个vector

        // 2.fuse the map
        // laserCloudCornerFromMap->clear();
        // laserCloudSurfFromMap->clear(); 
        laserCloudDeskewedFromMap->clear();//清理缓存！！！important
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            // *laserCloudSurfFromMap   += laserCloudSurfSurroundingVec[i];
            *laserCloudDeskewedFromMap += laserCloudDeskewedSurroundingVec[i];
        }

        // 3.分别对Corner和Surface特征进行采样
        
        // // Downsample the surrounding corner key frames (or map)
        // downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        // downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        // // Downsample the surrounding surf key frames (or map)
        // downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        // downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);

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
    // 保存地图线程
    void saveMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            // publishGlobalMap(); // 发布 全局地图点云(1000m以内)
        }

        // 以pcd格式保存地图
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ...123" << endl;
        // 1.create directory and remove old files, 删除文件夹再重建!!!
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str()); ++unused;
        
        pcl::io::savePCDFileASCII(savePCDDirectory + "RGB_Map.pcd", *outRGBCloud); // 所有RGB特征点云
        cout << "Saving map to pcd files completed🍎" << endl;
    }

    /**
     * @brief 从相机系的一个点得到去畸变的像素坐标投影点
     * 
     * @param point 相机系下某点
     * @param u 
     * @param v 
     */
    void cameraProjective(const PointType &point, double &u, double &v) {
        double mx_d, my_d,mx2_d, mxy_d, my2_d;
        double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d;
        //归一化坐标,
        double nx = point.x / point.z;
        double ny = point.y / point.z;
        mx_d = nx;
        my_d = ny;
        // 进行去畸变，参考14讲94页
        mx2_d = mx_d*mx_d;
        my2_d = my_d*my_d;

        mxy_d = mx_d*my_d;
        rho2_d = mx2_d+my2_d;
        rho4_d = rho2_d*rho2_d;
        radDist_d = 1 + k1*rho2_d+k2*rho4_d;
        Dx_d = nx*radDist_d + p2*(rho2_d+2*mx2_d) + 2*p1*mxy_d;
        Dy_d = ny*radDist_d + p1*(rho2_d+2*my2_d) + 2*p2*mxy_d;
        
        u = fx * Dx_d + cx;
        v = fy * Dy_d + cy;
    }
};






int main(int argc, char** argv)
{
    ros::init(argc, argv, "RGB");

    RGB rgb;

    ROS_INFO("\033[1;32m----> Lidar Map Optimization Started.\033[0m");
    
    std::thread saveMap_Thread(&RGB::saveMapThread, &rgb);
    ros::spin();
    saveMap_Thread.join();

    return 0;
}




