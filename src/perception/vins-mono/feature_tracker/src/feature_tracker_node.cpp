#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::msg::Image::ConstSharedPtr> img_buf;

rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_img;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match;
rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;
rclcpp::Node::SharedPtr node;

template <typename T>
double toSec(const T &time) {
    return time.sec + time.nanosec / 1e9;
}

void img_callback(sensor_msgs::msg::CompressedImage::ConstSharedPtr img_msg)
{
    cv::Mat img = cv::imdecode(cv::Mat(img_msg->data), cv::IMREAD_COLOR);
    if (img.empty())
    {
        RCLCPP_ERROR(rclcpp::get_logger("CompressedImage"), "Failed to decode image!");
        return;
    }

    if (first_image_flag)
    {
        first_image_flag = false;
        first_image_time = toSec(img_msg->header.stamp);
        last_image_time = toSec(img_msg->header.stamp);
        return;
    }

    // Detect unstable camera stream
    if (toSec(img_msg->header.stamp) - last_image_time > 1.0 || toSec(img_msg->header.stamp) < last_image_time)
    {
        RCLCPP_WARN(node->get_logger(), "%s image discontinue! reset the feature tracker!", AT);
        first_image_flag = true;
        last_image_time = 0;
        pub_count = 1;
        std_msgs::msg::Bool restart_flag;
        restart_flag.data = true;
        pub_restart->publish(restart_flag);
        return;
    }
    last_image_time = toSec(img_msg->header.stamp);

    // Frequency control
    if (round(1.0 * pub_count / (toSec(img_msg->header.stamp) - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        if (abs(1.0 * pub_count / (toSec(img_msg->header.stamp) - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = toSec(img_msg->header.stamp);
            pub_count = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    cv_bridge::CvImagePtr ptr(new cv_bridge::CvImage);
    ptr->header = img_msg->header;
    ptr->image = img;
    ptr->encoding = sensor_msgs::image_encodings::BGR8;

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        RCLCPP_DEBUG(node->get_logger(), "%s processing camera %d", AT, i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), toSec(img_msg->header.stamp));
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

    if (PUB_THIS_FRAME)
    {
        pub_count++;
        sensor_msgs::msg::PointCloud::SharedPtr feature_points(new sensor_msgs::msg::PointCloud);
        sensor_msgs::msg::ChannelFloat32 id_of_point;
        sensor_msgs::msg::ChannelFloat32 u_of_point;
        sensor_msgs::msg::ChannelFloat32 v_of_point;
        sensor_msgs::msg::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::msg::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::msg::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        RCLCPP_DEBUG(node->get_logger(), "%s publish %f, at %f", AT, toSec(feature_points->header.stamp), rclcpp::Clock().now().nanoseconds() / 1e9);
        // skip the first image; since no optical speed on first image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img->publish(*feature_points);

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                }
            }

            sensor_msgs::msg::CompressedImage::SharedPtr compressed_msg(new sensor_msgs::msg::CompressedImage);
            compressed_msg->header = img_msg->header;
            compressed_msg->format = "jpeg";
            std::vector<uchar> buf;
            cv::imencode(".jpg", ptr->image, buf);
            compressed_msg->data = std::move(buf);

            pub_match->publish(*compressed_msg);
        }
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    node = rclcpp::Node::make_shared("feature_tracker");
    // rcutils_logging_set_logger_level(node->get_logger().get_name(), RCUTILS_LOG_SEVERITY_DEBUG);
    readParameters(node);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                RCLCPP_INFO(node->get_logger(), "%s load mask fail", AT);
            }
            else
                RCLCPP_INFO(node->get_logger(), "%s load mask success", AT);
        }
    }

    //auto sub_img = node->create_subscription<sensor_msgs::msg::Image>(IMAGE_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), img_callback);
    auto sub_img = node->create_subscription<sensor_msgs::msg::CompressedImage>(IMAGE_TOPIC,rclcpp::QoS(rclcpp::KeepLast(100)),img_callback);

    pub_img = node->create_publisher<sensor_msgs::msg::PointCloud>("/feature_tracker/feature", 1000);
    pub_match = node->create_publisher<sensor_msgs::msg::Image>("/feature_tracker/feature_img",1000);
    pub_restart = node->create_publisher<std_msgs::msg::Bool>("/feature_tracker/restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    rclcpp::spin(node);
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?