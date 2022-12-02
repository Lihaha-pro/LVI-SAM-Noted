/*
 * @Author: lhli14 lhli14@iflytek.com
 * @Date: 2022-11-01 17:36:42
 * @LastEditors: lhli14 lhli14@iflytek.com
 * @LastEditTime: 2022-11-30 15:27:42
 * @FilePath: /LVI-SAM-Noted/src/lidar_odometry/renderRGB.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#pragma once
#include <atomic>
#include <unordered_set>
#include "opencv2/opencv.hpp"
#include<cv_bridge/cv_bridge.h>
#include "utility.h"
#include "lvi_sam/cloud_info.h"
#include <queue>
#include <utility>
#include <iostream>


cv::RNG g_rng = cv::RNG(0);
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

//全局地图类型
template <typename data_type = float, typename T = void *>
class Hash_map_3d
{
public:
    using hash_3d_T = std::unordered_map<data_type, std::unordered_map<data_type, std::unordered_map<data_type, T>>>;
    hash_3d_T m_map_3d_hash_map;
    void insert(const data_type &x, const data_type &y, const data_type &z, const T &target)
    {
        m_map_3d_hash_map[x][y][z] = target;
    }
    /**
     * @brief 判断是否已经存在体素，没有返回0
     * 
     */
    int if_exist(const data_type &x, const data_type &y, const data_type &z)
    {
        if(m_map_3d_hash_map.find(x) == m_map_3d_hash_map.end()  )
        {
            return 0;
        }
        else if(m_map_3d_hash_map[x].find(y) ==  m_map_3d_hash_map[x].end() )
        {
            return 0;
        }
        else if( m_map_3d_hash_map[x][y].find(z) == m_map_3d_hash_map[x][y].end() )
        {
            return 0;
        }
        return 1;
    }

    void clear()
    {
        m_map_3d_hash_map.clear();
    }

    int total_size()
    {
        int count =0 ;
        for(auto it : m_map_3d_hash_map)
        {
            for(auto it_it: it.second)
            {
                for( auto it_it_it: it_it.second )
                {
                    count++;
                }
            }
        }
        return count;
    }
};

//全局地图点数据结构
class RGB_pts
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#if 0
    std::atomic<double> m_pos[3];
    std::atomic<double> m_rgb[3];
    std::atomic<double> m_cov_rgb[3];
    std::atomic<double> m_gray;
    std::atomic<double> m_cov_gray;
    std::atomic<int> m_N_gray;
    std::atomic<int> m_N_rgb;
#else
    double m_pos[ 3 ] = { 0 };
    double m_rgb[ 3 ] = { 0 };
    double m_cov_rgb[ 3 ] = { 0 };
    double m_gray = 0;
    double m_cov_gray = 0;
    int    m_N_gray = 0;
    int    m_N_rgb = 0;//观测到该点的次数
    int    m_pt_index = 0;
#endif
    Eigen::Matrix< double, 2, 1>      m_img_vel;
    Eigen::Matrix< double, 2, 1>      m_img_pt_in_last_frame;
    Eigen::Matrix< double, 2, 1>      m_img_pt_in_current_frame;
    int        m_is_out_lier_count = 0;
    cv::Scalar m_dbg_color;
    double     m_obs_dis = 0;
    double     m_last_obs_time = 0;
    void clear()
    {
        m_rgb[ 0 ] = 0;
        m_rgb[ 1 ] = 0;
        m_rgb[ 2 ] = 0;
        m_gray = 0;
        m_cov_gray = 0;
        m_N_gray = 0;
        m_N_rgb = 0;
        m_obs_dis = 0;
        m_last_obs_time = 0;
        int r = g_rng.uniform( 0, 256 );
        int g = g_rng.uniform( 0, 256 );
        int b = g_rng.uniform( 0, 256 );
        m_dbg_color = cv::Scalar( r, g, b );
        // m_rgb = vec_3(255, 255, 255);
    };

    RGB_pts()
    {
        // m_pt_index = g_pts_index++;
        clear();
    };
    ~RGB_pts(){};

    void set_pos( const Eigen::Matrix< double, 3, 1> &pos ) {
        m_pos[0] = pos[0];
        m_pos[1] = pos[1];
        m_pos[2] = pos[2];
    }
    Eigen::Matrix< double, 3, 1> get_pos() {
        return Eigen::Matrix<double, 3, 1>(m_pos[0], m_pos[1], m_pos[2]);
    }
    Eigen::Matrix< double, 3, 1> get_rgb() {
        return Eigen::Matrix<double, 3, 1>(m_rgb[0], m_rgb[1], m_rgb[2]);
    }
    Eigen::Matrix< double, 3, 3> get_rgb_cov() {
        Eigen::Matrix<double, 3, 3> cov_mat = Eigen::Matrix<double, 3, 3>::Zero();
        for (int i = 0; i < 3; i++) {
            cov_mat(i, i) = m_cov_rgb[i];
        }
        return cov_mat;
    }
    pcl::PointXYZI get_pt() {
        pcl::PointXYZI pt;
        pt.x = m_pos[0];
        pt.y = m_pos[1];
        pt.z = m_pos[2];
        return pt;
    }
    void update_gray( const double gray, double obs_dis = 1.0 ) {
        if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2))
        {
            return;
        }
        m_gray = (m_gray * m_N_gray + gray) / (m_N_gray + 1);
        if (m_obs_dis == 0 || (obs_dis < m_obs_dis))
        {
            m_obs_dis = obs_dis;
            // m_gray = gray;
        }
        m_N_gray++;
        // TODO: cov update
    }

    /**
     * @brief 
     * 
     * @param rgb 点的RGB颜色
     * @param obs_dis 该点到相机的距离
     * @param obs_sigma 观测方差
     * @param obs_time 观测到这个点相机的时间
     * @return int 0没有更新 1更新
     */
    int update_rgb( const Eigen::Matrix< double, 3, 1> &rgb, const double obs_dis, const Eigen::Matrix< double, 3, 1> obs_sigma, const double obs_time ) {
        
        if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2))//之前观测过并且本次观测距离太远，就跳过
        {
            //m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2)
            return 0;
        }
        // 第一次观测
        if( m_N_rgb == 0)
        {
            // For first time of observation.
            m_last_obs_time = obs_time; //更新时间
            m_obs_dis = obs_dis; //更新距离
            for (int i = 0; i < 3; i++)
            {
                m_rgb[i] = rgb[i];//颜色
                m_cov_rgb[i] = obs_sigma(i) ;//方差
            }
            m_N_rgb = 1;
            return 0;
        }
        // State estimation for robotics, section 2.2.6, page 37-38
        // 不是第一次观测
        for(int i = 0; i < 3; i++)
        {
            // 更新方差
            m_cov_rgb[i] = (m_cov_rgb[i] + 0.1 * (obs_time - m_last_obs_time)); // Add process noise
            double old_sigma = m_cov_rgb[i];
            m_cov_rgb[i] = sqrt( 1.0 / (1.0 / m_cov_rgb[i] / m_cov_rgb[i] + 1.0 / obs_sigma(i) / obs_sigma(i)) );
            // 更新RGB，前后观测进行融合
            m_rgb[i] = m_cov_rgb[i] * m_cov_rgb[i] * ( m_rgb[i] / old_sigma / old_sigma + rgb(i) / obs_sigma(i) / obs_sigma(i) );
        }

        if (obs_dis < m_obs_dis)
        {
            m_obs_dis = obs_dis;
        }
        m_last_obs_time = obs_time;
        m_N_rgb++;
        return 1;
    }

  private:
    // friend class boost::serialization::access;
    template < typename Archive >
    void serialize( Archive &ar, const unsigned int version )
    {
        ar &m_pos;
        ar &m_rgb;
        ar &m_pt_index;
        ar &m_cov_rgb;
        ar &m_gray;
        ar &m_N_rgb;
        ar &m_N_gray;
    }
};
using RGB_pt_ptr = std::shared_ptr< RGB_pts >;