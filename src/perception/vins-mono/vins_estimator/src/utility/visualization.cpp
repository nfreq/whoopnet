#include "visualization.h"

using namespace Eigen;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_latest_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_relo_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_point_cloud;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_margin_cloud;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_relo_relative_pose;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
nav_msgs::msg::Path path, relo_path;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;

CameraPoseVisualization cameraposevisual(0, 1, 0, 1);
CameraPoseVisualization keyframebasevisual(0.0, 0.0, 1.0, 1.0);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

void registerPub(const rclcpp::Node::SharedPtr &n)
{
    rclcpp::QoS qos_besteffort = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default))
                            .reliability(rclcpp::ReliabilityPolicy::BestEffort)
                            .history(rclcpp::HistoryPolicy::KeepLast)
                            .keep_last(100);
    pub_latest_odometry = n->create_publisher<nav_msgs::msg::Odometry>("whoopnet/perception/vins_mono/vins_estimator/imu_propagate", qos_besteffort);
    pub_path = n->create_publisher<nav_msgs::msg::Path>("whoopnet/perception/vins_mono/vins_estimator/path", qos_besteffort);
    pub_relo_path = n->create_publisher<nav_msgs::msg::Path>("whoopnet/perception/vins_mono/vins_estimator/relocalization_path", qos_besteffort);
    pub_odometry = n->create_publisher<nav_msgs::msg::Odometry>("whoopnet/perception/vins_mono/vins_estimator/odometry", qos_besteffort);
    pub_point_cloud = n->create_publisher<sensor_msgs::msg::PointCloud2>("whoopnet/perception/vins_mono/vins_estimator/point_cloud", qos_besteffort);
    pub_margin_cloud = n->create_publisher<sensor_msgs::msg::PointCloud2>("whoopnet/perception/vins_mono/vins_estimator/history_cloud", qos_besteffort);
    pub_key_poses = n->create_publisher<visualization_msgs::msg::Marker>("whoopnet/perception/vins_mono/vins_estimator/key_poses", qos_besteffort);
    pub_camera_pose = n->create_publisher<nav_msgs::msg::Odometry>("whoopnet/perception/vins_mono/vins_estimator/camera_pose", qos_besteffort);
    pub_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("whoopnet/perception/vins_mono/vins_estimator/camera_pose_visual", qos_besteffort);
    pub_keyframe_pose = n->create_publisher<nav_msgs::msg::Odometry>("whoopnet/perception/vins_mono/vins_estimator/keyframe_pose", qos_besteffort);
    pub_keyframe_point = n->create_publisher<sensor_msgs::msg::PointCloud>("whoopnet/perception/vins_mono/vins_estimator/keyframe_point", qos_besteffort);
    pub_extrinsic = n->create_publisher<nav_msgs::msg::Odometry>("whoopnet/perception/vins_mono/vins_estimator/extrinsic", qos_besteffort);
    pub_relo_relative_pose=  n->create_publisher<nav_msgs::msg::Odometry>("whoopnet/perception/vins_mono/vins_estimator/relo_relative_pose", qos_besteffort);

    cameraposevisual.setScale(1);
    cameraposevisual.setLineWidth(0.05);
    keyframebasevisual.setScale(0.1);
    keyframebasevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::msg::Header &header)
{
    Eigen::Quaterniond quadrotor_Q = Q ;

    nav_msgs::msg::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry->publish(odometry);
}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    RCLCPP_DEBUG_STREAM(node->get_logger(), AT << " position: " << estimator.Ps[WINDOW_SIZE].transpose());
    RCLCPP_DEBUG_STREAM(node->get_logger(), AT << " orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        //RCLCPP_DEBUG(node->get_logger(), "%s calibration result for camera %d", AT, i);
        RCLCPP_DEBUG_STREAM(node->get_logger(), AT << " extirnsic tic: " << estimator.tic[i].transpose());
        RCLCPP_DEBUG_STREAM(node->get_logger(), AT << " extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());
        if (ESTIMATE_EXTRINSIC)
        {
            cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
            Eigen::Matrix3d eigen_R;
            Eigen::Vector3d eigen_T;
            eigen_R = estimator.ric[i];
            eigen_T = estimator.tic[i];
            cv::Mat cv_R, cv_T;
            cv::eigen2cv(eigen_R, cv_R);
            cv::eigen2cv(eigen_T, cv_T);
            fs << "extrinsicRotation" << cv_R << "extrinsicTranslation" << cv_T;
            fs.release();
        }
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    RCLCPP_DEBUG(node->get_logger(), "%s vo solver costs: %f ms", AT, t);
    RCLCPP_DEBUG(node->get_logger(), "%s average of time %f ms", AT, sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    RCLCPP_DEBUG(node->get_logger(), "%s sum of path %f", AT, sum_of_path);
    if (ESTIMATE_TD)
        RCLCPP_INFO(node->get_logger(), "%s td %f", AT, estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        pub_odometry->publish(odometry);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path->publish(path);

        Vector3d correct_t;
        Vector3d correct_v;
        Quaterniond correct_q;
        correct_t = estimator.drift_correct_r * estimator.Ps[WINDOW_SIZE] + estimator.drift_correct_t;
        correct_q = estimator.drift_correct_r * estimator.Rs[WINDOW_SIZE];
        odometry.pose.pose.position.x = correct_t.x();
        odometry.pose.pose.position.y = correct_t.y();
        odometry.pose.pose.position.z = correct_t.z();
        odometry.pose.pose.orientation.x = correct_q.x();
        odometry.pose.pose.orientation.y = correct_q.y();
        odometry.pose.pose.orientation.z = correct_q.z();
        odometry.pose.pose.orientation.w = correct_q.w();

        pose_stamped.pose = odometry.pose.pose;
        relo_path.header = header;
        relo_path.header.frame_id = "world";
        relo_path.poses.push_back(pose_stamped);
        pub_relo_path->publish(relo_path);

        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(0);
        foutC << Utility::toSec(header.stamp) << ",";
        foutC.precision(5);
        foutC << estimator.Ps[WINDOW_SIZE].x() << ","
              << estimator.Ps[WINDOW_SIZE].y() << ","
              << estimator.Ps[WINDOW_SIZE].z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.Vs[WINDOW_SIZE].x() << ","
              << estimator.Vs[WINDOW_SIZE].y() << ","
              << estimator.Vs[WINDOW_SIZE].z() << "," << endl;
        foutC.close();
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::msg::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::msg::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = rclcpp::Duration(0, 0);

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::msg::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses->publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose->publish(odometry);

        cameraposevisual.reset();

        // For Testing, Disable Translation (Avoid drifting issues and validate orientation only)
        Vector3d PZ = Vector3d(0.0, 0.0, 0.0);
        cameraposevisual.add_pose(PZ, R);
        //cameraposevisual.add_pose(P, R);
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    // Main point cloud
    sensor_msgs::msg::PointCloud2 point_cloud;
    point_cloud.header = header;
    point_cloud.height = 1; // Unordered point cloud
    point_cloud.is_dense = false;

    // Initialize PointCloud2Modifier
    sensor_msgs::PointCloud2Modifier modifier(point_cloud);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgba"); // Add xyz and rgba fields
    modifier.resize(0); // Start with no points

    // Populate point_cloud
    for (auto &&it_per_id : estimator.f_manager.feature)
    {
        int used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;

        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        // Add point data
        modifier.resize(modifier.size() + 1); // Add space for one point
        sensor_msgs::PointCloud2Iterator<float> iter_x(point_cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(point_cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(point_cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint32_t> iter_rgba(point_cloud, "rgba");

        iter_x[modifier.size() - 1] = w_pts_i(0);
        iter_y[modifier.size() - 1] = w_pts_i(1);
        iter_z[modifier.size() - 1] = w_pts_i(2);

        // Set color (e.g., white: 0xFFFFFFFF)
        iter_rgba[modifier.size() - 1] = 0xFFFFFFFF; // RGBA (white color)
    }

    pub_point_cloud->publish(point_cloud);

    // Margin point cloud
    sensor_msgs::msg::PointCloud2 margin_cloud;
    margin_cloud.header = header;
    margin_cloud.height = 1; // Unordered point cloud
    margin_cloud.is_dense = false;

    // Initialize PointCloud2Modifier for margin cloud
    sensor_msgs::PointCloud2Modifier margin_modifier(margin_cloud);
    margin_modifier.setPointCloud2FieldsByString(2, "xyz", "rgba"); // Add xyz and rgba fields
    margin_modifier.resize(0); // Start with no points

    // Populate margin_cloud
    for (auto &&it_per_id : estimator.f_manager.feature)
    {
        int used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 &&
            it_per_id.solve_flag == 1)
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

            // Add the point to the PointCloud2 message
            margin_modifier.resize(margin_modifier.size() + 1); // Add space for one point
            sensor_msgs::PointCloud2Iterator<float> iter_x(margin_cloud, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(margin_cloud, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(margin_cloud, "z");
            sensor_msgs::PointCloud2Iterator<uint32_t> iter_rgba(margin_cloud, "rgba");

            // Populate fields
            iter_x[margin_modifier.size() - 1] = w_pts_i(0);
            iter_y[margin_modifier.size() - 1] = w_pts_i(1);
            iter_z[margin_modifier.size() - 1] = w_pts_i(2);

            // Set color (example: red for margined points)
            iter_rgba[margin_modifier.size() - 1] = 0xFF0000FF; // RGBA (red color)
        }
    }

    // Publish the populated PointCloud2
    pub_margin_cloud->publish(margin_cloud);
}


void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;

    geometry_msgs::msg::TransformStamped transform;
    tf2::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;

    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    transform.header.stamp = header.stamp;
    transform.header.frame_id = "world";
    transform.child_frame_id = "body";

    transform.transform.translation.x = correct_t(0);
    transform.transform.translation.y = correct_t(1);
    transform.transform.translation.z = correct_t(2);

    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    transform.transform.rotation.w = q.w();

    tf_broadcaster->sendTransform(transform);

    // camera frame
    transform.header.stamp = header.stamp;
    transform.header.frame_id = "body";
    transform.child_frame_id = "camera";

    transform.transform.translation.x = estimator.tic[0].x();
    transform.transform.translation.y = estimator.tic[0].y();
    transform.transform.translation.z = estimator.tic[0].z();

    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());

    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    transform.transform.rotation.w = q.w();

    tf_broadcaster->sendTransform(transform);

    nav_msgs::msg::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();

    pub_extrinsic->publish(odometry);
}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);

        nav_msgs::msg::Odometry odometry;
        odometry.header = estimator.Headers[WINDOW_SIZE - 2];
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose->publish(odometry);


        sensor_msgs::msg::PointCloud point_cloud;
        point_cloud.header = estimator.Headers[WINDOW_SIZE - 2];
        for (auto &&it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                      + estimator.Ps[imu_i];
                geometry_msgs::msg::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::msg::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point->publish(point_cloud);
    }
}

void pubRelocalization(const Estimator &estimator)
{
    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = rclcpp::Time(estimator.relo_frame_stamp);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.relo_relative_t.x();
    odometry.pose.pose.position.y = estimator.relo_relative_t.y();
    odometry.pose.pose.position.z = estimator.relo_relative_t.z();
    odometry.pose.pose.orientation.x = estimator.relo_relative_q.x();
    odometry.pose.pose.orientation.y = estimator.relo_relative_q.y();
    odometry.pose.pose.orientation.z = estimator.relo_relative_q.z();
    odometry.pose.pose.orientation.w = estimator.relo_relative_q.w();
    odometry.twist.twist.linear.x = estimator.relo_relative_yaw;
    odometry.twist.twist.linear.y = estimator.relo_frame_index;

    pub_relo_relative_pose->publish(odometry);
}