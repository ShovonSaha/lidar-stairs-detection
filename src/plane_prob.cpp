#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <vector>
#include <sstream>

#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/distances.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/bilateral.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/surface/mls.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

#include <visualization_msgs/Marker.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>






// ROS Publishers
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_axis_downsampling;
ros::Publisher pub_after_mls;
ros::Publisher pub_after_rotation;
ros::Publisher pub_after_plane_1;
ros::Publisher pub_after_plane_2;
ros::Publisher pub_after_plane_3;
ros::Publisher pub_after_plane_4;
ros::Publisher marker_pub;
ros::Publisher pub_after_plane_segmentation;

// Declare plane_coefficients globally
// std::vector<pcl::ModelCoefficients> plane_coefficients;
// std::vector<pcl::ModelCoefficients> plane_coefficients(4);  // Assuming you want to find 4 planes

std::vector<ros::Publisher> plane_publishers; // Publishers for segmented planes
std::vector<pcl::ModelCoefficients> plane_coefficients; // To story.e coefficients of planes

ros::Publisher getPlanePublisher(size_t plane_index, ros::NodeHandle& nh)
{
    // Modify the topic name based on your naming convention
    std::string topic_name = "/plane_" + std::to_string(plane_index);

    // Create and return the publisher
    return nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1);
}

// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg)
{
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}

// void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const std::string& frame_id) {
//     sensor_msgs::PointCloud2 output_msg;
//     pcl::toROSMsg(*cloud, output_msg);
//     output_msg.header.frame_id = frame_id;
//     output_msg.header.stamp = ros::Time::now();
//     publisher.publish(output_msg);
// }

// Function to publish a segmented plane as a marker
// void publishSegmentedPlaneMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& segmented_plane, const ros::Publisher& marker_publisher, const pcl::ModelCoefficients::Ptr& coefficients)
// {
//     // Create a marker for the segmented plane
//     visualization_msgs::Marker plane_marker;
//     plane_marker.header.frame_id = segmented_plane->header.frame_id;  // Assuming the cloud's frame is relevant
//     plane_marker.header.stamp = ros::Time::now();
//     plane_marker.ns = "segmented_plane";
//     plane_marker.id = marker_publisher.getNumSubscribers();  // Use the number of subscribers as an ID
//     plane_marker.type = visualization_msgs::Marker::CUBE;  // Use CUBE for a marker representing planes
//     plane_marker.action = visualization_msgs::Marker::ADD;

//     // Calculate the centroid of the segmented plane
//     Eigen::Vector4f centroid;
//     pcl::compute3DCentroid(*segmented_plane, centroid);
//     geometry_msgs::Point centroid_point;
//     centroid_point.x = centroid[0];
//     centroid_point.y = centroid[1];
//     centroid_point.z = centroid[2];

//     // Set the marker properties
//     plane_marker.pose.position = centroid_point;  // Set the position to the centroid
//     plane_marker.scale.x = 0.2;
//     plane_marker.scale.y = 0.7;
//     plane_marker.scale.z = 0.02;
//     plane_marker.color.a = 0.4;   // Opacity
//     plane_marker.color.r = 1.0;
//     plane_marker.color.g = 0.0;
//     plane_marker.color.b = 0.0;

//     // Calculate the orientation based on the plane's normal vector
//     Eigen::Vector3d normal_vector(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
//     normal_vector.normalize();  // Ensure the normal vector is normalized

//     // // Create a quaternion from the axis-angle representation
//     // Eigen::AngleAxisd rotation(normal_vector.cross(Eigen::Vector3d::UnitZ()), std::acos(normal_vector.dot(Eigen::Vector3d::UnitZ())));
//     // Eigen::Quaterniond quat(rotation);
    
//     // Create a quaternion from the axis-angle representation
//     Eigen::Quaterniond quat(Eigen::AngleAxisd(std::acos(normal_vector.dot(Eigen::Vector3d::UnitZ())), normal_vector.cross(Eigen::Vector3d::UnitZ())));


//     // Set the orientation using the quaternion
//     plane_marker.pose.orientation.x = quat.x();
//     plane_marker.pose.orientation.y = quat.y();
//     plane_marker.pose.orientation.z = quat.z();
//     plane_marker.pose.orientation.w = quat.w();

//     // Publish the marker
//     marker_publisher.publish(plane_marker);

//     // Print the equation of the plane
//     // ROS_INFO("Equation of Plane %u: Ax + By + Cz + D = 0", marker_publisher.getNumSubscribers());
//     // ROS_INFO("A: %f, B: %f, C: %f, D: %f", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
// }







// void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg) {
//     sensor_msgs::PointCloud2 output_msg;
//     pcl::toROSMsg(*cloud, output_msg);
//     output_msg.header = original_msg->header; // Use the frame_id from the original message
//     publisher.publish(output_msg);
// }

void publishSegmentedPlaneMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& segmented_plane, const ros::Publisher& marker_publisher, const pcl::ModelCoefficients::Ptr& coefficients, int id) {
    visualization_msgs::Marker plane_marker;
    plane_marker.header.frame_id = segmented_plane->header.frame_id; // Assuming the cloud's frame is used
    plane_marker.header.stamp = ros::Time::now();
    plane_marker.ns = "segmented_plane";
    plane_marker.id = id;
    plane_marker.type = visualization_msgs::Marker::CUBE;
    plane_marker.action = visualization_msgs::Marker::ADD;

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*segmented_plane, centroid);
    plane_marker.pose.position.x = centroid[0];
    plane_marker.pose.position.y = centroid[1];
    plane_marker.pose.position.z = centroid[2];
    plane_marker.scale.x = 1.0; // Adjust based on your needs
    plane_marker.scale.y = 1.0;
    plane_marker.scale.z = 0.01; // Thin plane
    plane_marker.color.a = 0.8; // Transparency
    plane_marker.color.r = 0.0;
    plane_marker.color.g = 1.0;
    plane_marker.color.b = 0.0;

    Eigen::Vector3d normal_vector(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    normal_vector.normalize();
    Eigen::Quaterniond quat(Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(0, 0, 1), normal_vector));

    plane_marker.pose.orientation.x = quat.x();
    plane_marker.pose.orientation.y = quat.y();
    plane_marker.pose.orientation.z = quat.z();
    plane_marker.pose.orientation.w = quat.w();

    marker_publisher.publish(plane_marker);
}











pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterY(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.7, 0.7);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
    pass.filter(*cloud_filtered_y);

    return cloud_filtered_y;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplingAlongAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& axis, double min_limit, double max_limit)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(0.05, 0.05, 0.05);  // Set an initial leaf size
    voxel_grid.setFilterFieldName(axis);
    voxel_grid.setFilterLimits(min_limit, max_limit);

    // Create a new point cloud to store the downsampled points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);

    // Apply voxel grid downsampling
    voxel_grid.filter(*cloud_downsampled);

    // Update the width and height fields of the downsampled point cloud
    cloud_downsampled->width = cloud_downsampled->size();
    cloud_downsampled->height = 1;

    return cloud_downsampled;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr applyMLSSmoothing(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal> mls_points;
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

    mls.setComputeNormals(false); // We don't need normals in this case
    mls.setInputCloud(cloud);
    mls.setPolynomialOrder(1);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(0.05); // Adjust based on your dataset

    mls.process(mls_points);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::copyPointCloud(mls_points, *cloud_smoothed);
    return cloud_smoothed;
}


// bool customRegionGrowing(const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
// {
//     Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap(), point_b_normal = point_b.getNormalVector3fMap();
//     if (squared_distance < 10000)
//     {
//         if (std::abs(point_a_normal.dot(point_b_normal)) > std::cos(30.0f / 180.0f * static_cast<float>(M_PI)))
//             return true;
//     }
//     return false;
// }


// Normal Estimation
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(50);  // Adjust the value based on your data
    ne.compute(*normals);

    return normals;
}


void segmentAndPublishPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, ros::NodeHandle& nh, const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);

    int i = 0;
    while (cloud->points.size() > 3) {
        pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setNormalDistanceWeight(0.06);
        seg.setMaxIterations(100);
        seg.setDistanceThreshold(0.02);
        seg.setInputCloud(cloud);
        seg.setInputNormals(cloud_normals);
        seg.segment(*inliers, *coefficients);


        if (inliers->indices.empty()) {
            ROS_WARN("No further planes could be segmented.");
            break;
        }

        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_plane);

        if (cloud_plane->points.empty()) {
            ROS_WARN("Plane extraction resulted in an empty cloud.");
            continue;
        }

        // Dynamically create publisher for each segmented plane
        std::stringstream ss;
        ss << "extracted_plane_" << i;
        ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(ss.str(), 1, true);
        plane_publishers.push_back(pub); // Keep track of the dynamically created publishers

        // Use the original message to preserve the header and frame_id
        publishProcessedCloud(cloud_plane, pub, msg);

        // Publish plane marker for visualization
        publishSegmentedPlaneMarker(cloud_plane, marker_pub, coefficients, i);

        ROS_INFO("Segmented plane %d: Published to %s", i, ss.str().c_str());

        // Remove the segmented plane from the cloud to continue segmentation
        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>());
        extract.filter(*cloud_temp);
        cloud.swap(cloud_temp);

        i++;
    }
}





// void segmentAndPublishPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, ros::NodeHandle& nh) {
//     int i = 0;
//     pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

//     ne.setSearchMethod(tree);
//     ne.setInputCloud(cloud);
//     ne.setKSearch(50);
//     ne.compute(*cloud_normals);

//     while (cloud->points.size() > 0.1 * cloud_normals->points.size()) {
//         pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
//         pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//         pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
//         pcl::ExtractIndices<pcl::PointXYZ> extract;

//         seg.setOptimizeCoefficients(true);
//         seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
//         seg.setMethodType(pcl::SAC_RANSAC);
//         seg.setNormalDistanceWeight(0.1);
//         seg.setMaxIterations(100);
//         seg.setDistanceThreshold(0.03);
//         seg.setInputCloud(cloud);
//         seg.setInputNormals(cloud_normals);
//         seg.segment(*inliers, *coefficients);

//         if (inliers->indices.size() == 0) {
//             break;
//         }

//         extract.setInputCloud(cloud);
//         extract.setIndices(inliers);
//         extract.setNegative(false);
//         extract.filter(*cloud_plane);

//         // Publish each plane to a separate topic
//         std::stringstream ss;
//         ss << "extracted_plane_" << i++;
//         ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(ss.str(), 1, true);
//         sensor_msgs::PointCloud2 output;
//         pcl::toROSMsg(*cloud_plane, output);
//         output.header.frame_id = "base_link"; // Change according to your frame
//         pub.publish(output);
        
//         // Remove the extracted plane from the input cloud
//         extract.setNegative(true);
//         pcl::PointCloud<pcl::PointXYZ> cloudF;
//         extract.filter(cloudF);
//         cloud.swap(boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(cloudF));
//     }
// }


// // Function to segment planes and publish segmented planes
// pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::ModelCoefficients>& plane_coefficients, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::copyPointCloud(*cloud, *remaining_cloud);

//     for (size_t i = 0; i < plane_coefficients.size(); ++i)
//     {
//         pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//         pcl::ModelCoefficients::Ptr current_coefficients(new pcl::ModelCoefficients);

//         pcl::SACSegmentation<pcl::PointXYZ> seg;
//         seg.setOptimizeCoefficients(true);
//         seg.setModelType(pcl::SACMODEL_PLANE);
//         seg.setMethodType(pcl::SAC_RANSAC);
//         seg.setDistanceThreshold(distance_threshold);
//         seg.setInputCloud(remaining_cloud);
//         seg.segment(*inliers, *current_coefficients);

//         if (inliers->indices.size() < 100)
//         {
//             // If the number of inliers is too small, skip this plane
//             continue;
//         }

//         pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::ExtractIndices<pcl::PointXYZ> extract;
//         extract.setInputCloud(remaining_cloud);
//         extract.setIndices(inliers);
//         extract.setNegative(false);
//         extract.filter(*current_plane);

//         // Publish processed cloud and segmented plane marker
//         publishProcessedCloud(current_plane, nh.advertise<sensor_msgs::PointCloud2>("/plane_" + std::to_string(i + 1), 1), original_msg);
//         publishSegmentedPlaneMarker(current_plane, marker_pub, current_coefficients);

//         extract.setNegative(true);
//         extract.filter(*remaining_cloud);

//         // Store the plane coefficients
//         plane_coefficients[i] = *current_coefficients;
//     }

//     return remaining_cloud;
// }


// // pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::ModelCoefficients>& plane_coefficients, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
// pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::ModelCoefficients>& plane_coefficients, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::copyPointCloud(*cloud, *remaining_cloud);

//     // Sort planes based on the number of inliers (largest plane first)
//     std::sort(plane_coefficients.begin(), plane_coefficients.end(), [](const pcl::ModelCoefficients& a, const pcl::ModelCoefficients& b) {
//         return a.values.size() > b.values.size();
//     });

//     for (size_t i = 0; i < plane_coefficients.size(); ++i)
//     {
//         pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//         pcl::ModelCoefficients::Ptr current_coefficients(new pcl::ModelCoefficients);

//         pcl::SACSegmentation<pcl::PointXYZ> seg;
//         seg.setOptimizeCoefficients(true);
//         seg.setModelType(pcl::SACMODEL_PLANE);
//         seg.setMethodType(pcl::SAC_RANSAC);
//         seg.setDistanceThreshold(distance_threshold);
//         seg.setInputCloud(remaining_cloud);
//         seg.segment(*inliers, *current_coefficients);

//         if (inliers->indices.size() < 100)
//         {
//             // If the number of inliers is too small, skip this plane
//             continue;
//         }

//         // Perform clustering
//         pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::ExtractIndices<pcl::PointXYZ> extract;
//         extract.setInputCloud(remaining_cloud);
//         extract.setIndices(inliers);
//         extract.setNegative(false);
//         extract.filter(*current_plane);

//         std::vector<pcl::PointIndices> cluster_indices;
//         pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//         // ec.setClusterTolerance(distance_threshold); // Adjust based on your point cloud characteristics
//         ec.setClusterTolerance(0.06);
//         ec.setMinClusterSize(50);    // Adjust based on your point cloud characteristics
//         // ec.setMaxClusterSize(current_plane->size());  // Adjust based on your point cloud characteristics
//         ec.setMaxClusterSize(200);
//         ec.setInputCloud(current_plane);
//         ec.extract(cluster_indices);

//         // Consider only the largest cluster
//         if (cluster_indices.size() > 0)
//         {
//             pcl::PointIndices::Ptr largest_cluster = boost::make_shared<pcl::PointIndices>(cluster_indices[0]);
//             extract.setIndices(largest_cluster);
//             extract.filter(*current_plane);

//             // Publish processed cloud and segmented plane marker
//             publishProcessedCloud(current_plane, nh.advertise<sensor_msgs::PointCloud2>("/plane_" + std::to_string(i + 1), 1), original_msg);
//             publishSegmentedPlaneMarker(current_plane, marker_pub, current_coefficients);

//             // Print the equation of the plane
//             ROS_INFO("Equation of Plane %zu: Ax + By + Cz + D = 0", i + 1);
//             ROS_INFO("A: %f, B: %f, C: %f, D: %f", current_coefficients->values[0], current_coefficients->values[1], current_coefficients->values[2], current_coefficients->values[3]);


//             extract.setNegative(true);
//             extract.filter(*remaining_cloud);

//             // Store the plane coefficients
//             plane_coefficients[i] = *current_coefficients;
//         }
//     }

//     return remaining_cloud;
// }




// pcl::PointCloud<pcl::PointXYZ>::Ptr refineSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& clustered_cloud, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr refined_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::copyPointCloud(*clustered_cloud, *refined_cloud);

//     for (size_t i = 0; i < refined_cloud->size(); ++i)
//     {
//         pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//         pcl::ModelCoefficients::Ptr current_coefficients(new pcl::ModelCoefficients);

//         pcl::SACSegmentation<pcl::PointXYZ> seg;
//         seg.setOptimizeCoefficients(true);
//         seg.setModelType(pcl::SACMODEL_PLANE);
//         seg.setMethodType(pcl::SAC_RANSAC);
//         seg.setDistanceThreshold(distance_threshold);
//         seg.setInputCloud(refined_cloud);
//         seg.segment(*inliers, *current_coefficients);

//         if (inliers->indices.size() < 100)
//         {
//             // If the number of inliers is too small, skip this plane
//             continue;
//         }

//         pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::ExtractIndices<pcl::PointXYZ> extract;
//         extract.setInputCloud(refined_cloud);
//         extract.setIndices(inliers);
//         extract.setNegative(false);
//         extract.filter(*current_plane);

//         // Publish processed cloud and segmented plane marker
//         publishProcessedCloud(current_plane, nh.advertise<sensor_msgs::PointCloud2>("/refined_plane_" + std::to_string(i + 1), 1), original_msg);
//         publishSegmentedPlaneMarker(current_plane, marker_pub, current_coefficients);

//         // Print the number of inliers
//         ROS_INFO("Number of inliers in refined plane %zu: %zu", i + 1, inliers->indices.size());

//         extract.setNegative(true);
//         extract.filter(*refined_cloud);

//         // Store the plane coefficients
//         // Assuming you want to store coefficients for each refined plane
//         plane_coefficients.push_back(*current_coefficients);
//     }

//     return refined_cloud;
// }


// pcl::PointCloud<pcl::PointXYZ>::Ptr euclideanClusterAndSegment(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double cluster_tolerance, int min_cluster_size, int max_cluster_size, double distance_threshold, const sensor_msgs::PointCloud2ConstPtr& original_msg, ros::NodeHandle& nh)
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::copyPointCloud(*cloud, *clustered_cloud);

//     // Euclidean Clustering
//     pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//     ec.setClusterTolerance(cluster_tolerance);
//     ec.setMinClusterSize(min_cluster_size);
//     ec.setMaxClusterSize(max_cluster_size);
//     ec.setInputCloud(clustered_cloud);
//     std::vector<pcl::PointIndices> cluster_indices;
//     ec.extract(cluster_indices);

//     // Iterate through clusters
//     for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
//     {
//         pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::ExtractIndices<pcl::PointXYZ> extract;
//         pcl::PointIndices::Ptr cluster_indices_ptr = boost::make_shared<pcl::PointIndices>(*it);
//         extract.setInputCloud(clustered_cloud);
//         extract.setIndices(cluster_indices_ptr);
//         extract.setNegative(false);
//         extract.filter(*current_cluster);

//         // Apply SACSegmentation on the cluster
//         refineSegmentation(current_cluster, distance_threshold, original_msg, nh);
//     }

//     return clustered_cloud;
// }

// bool customCondition(const pcl::PointXYZ& point_a, const pcl::PointXYZ& point_b, float squared_distance, const pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals, int idx_a, int idx_b, float angle_threshold) {
//     Eigen::Vector3f normal_a = cloud_normals->points[idx_a].getNormalVector3fMap();
//     Eigen::Vector3f normal_b = cloud_normals->points[idx_b].getNormalVector3fMap();
    
//     float dot_product = normal_a.dot(normal_b);
//     float angle = acos(dot_product); // Angle in radians
    
//     // Check if the angle is less than the threshold
//     return angle < angle_threshold;
// }

std::vector<pcl::PointIndices> conditionalEuclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals) {
    pcl::ConditionalEuclideanClustering<pcl::PointXYZ> cec(true);
    cec.setInputCloud(cloud);

    // Lambda function for custom clustering condition
    cec.setConditionFunction([cloud, cloud_normals](const pcl::PointXYZ& point_a, const pcl::PointXYZ& point_b, float squared_distance) -> bool {
        int idx_a = &point_a - &cloud->points[0];
        int idx_b = &point_b - &cloud->points[0];
        
        Eigen::Vector3f normal_a = cloud_normals->points[idx_a].getNormalVector3fMap();
        Eigen::Vector3f normal_b = cloud_normals->points[idx_b].getNormalVector3fMap();
        
        float dot_product = normal_a.dot(normal_b);
        float angle = acos(std::min(std::max(dot_product, -1.0f), 1.0f)); // Ensure the value is within [-1, 1] for acos
        
        // Define your threshold angle in radians, e.g., 10 degrees
        float angle_threshold = 10.0 * M_PI / 180.0;
        
        // Condition based on the angle between normals
        return angle < angle_threshold;
    });

    cec.setClusterTolerance(0.8); // 2cm tolerance in Euclidean distance
    cec.setMinClusterSize(50); // Minimum cluster size
    cec.setMaxClusterSize(400); // Maximum cluster size
    
    std::vector<pcl::PointIndices> clusters;
    cec.segment(clusters);

    return clusters;
}



void publishClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& clusters, ros::NodeHandle& nh) {
    int i = 0;
    for (const auto& cluster : clusters) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : cluster.indices) {
            cluster_cloud->points.push_back(cloud->points[idx]);
        }
        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cluster_cloud, output);
        output.header.frame_id = cloud->header.frame_id;  // Assuming the cloud's frame is relevant

        std::stringstream ss;
        ss << "cluster_" << i++;
        ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(ss.str(), 1, true);
        pub.publish(output);
    }
}



// Function to calculate the variance of a plane
double calculatePlaneVariance(const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane)
{
    // Calculate the centroid of the plane
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*plane, centroid);

    // Calculate the variance
    double variance = 0.0;
    for (const auto& point : plane->points)
    {
        double distance = (centroid.head<3>() - point.getVector3fMap().head<3>()).norm();
        variance += distance * distance;
    }
    return variance / plane->size();
}

// Main callback function for processing PointCloud2 messages
void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg, ros::NodeHandle& nh)
{
    // Convert ROS PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Initial processing steps here (e.g., passthrough filtering, downsampling and )
    // Passthrough Filtering with Y-Axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud);
    publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, input_msg);

    // Downsampling along X-axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(cloud_after_passthrough_y, "x", 0.0, 2.5);
    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

    // Moving Least Squares (MLS) Surface Smoothing and Reconstruction
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_smoothing = applyMLSSmoothing(cloud_after_axis_downsampling);
    publishProcessedCloud(cloud_after_smoothing, pub_after_mls, input_msg);


    // Normal Estimation
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_downsampled = computeNormals(cloud_after_axis_downsampling);

    // Conditional Euclidean Clustering
    auto clusters = conditionalEuclideanClustering(cloud_after_axis_downsampling, cloud_normals_downsampled);
    // Publish clusters
    publishClusters(cloud_after_axis_downsampling, clusters, nh);

    // Step 3: Rotation based on tilt angle
    // float tilt_angle = -15.0; // Example tilt angle in degrees
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated(new pcl::PointCloud<pcl::PointXYZ>);
    // Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    // transform.rotate(Eigen::AngleAxisf(tilt_angle * M_PI / 180.0, Eigen::Vector3f::UnitY()));
    // pcl::transformPointCloud(*cloud_after_axis_downsampling, *cloud_rotated, transform);

    // publishProcessedCloud(cloud_rotated, pub_after_rotation, input_msg);

    // // Normal Estimation for rotated pointcloud
    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_after_rotation = computeNormals(cloud_rotated);
    
    // Update the number of planes based on the size of plane_coefficients
    // size_t num_planes = plane_coefficients.size();
    
    // Step 2: Segment planes
    
    // segmentAndPublishPlanes(cloud, nh);

    segmentAndPublishPlanes(cloud_after_axis_downsampling, nh, input_msg);

    // Assuming you want to find 4 planes
    // size_t num_planes = 4;
    // double distance_threshold = 0.05;
    
    // parameters passed: segmentPlanes(input pointcloud, plane_coefficients, distance threshold, sensor msg, node handle)
    // pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud = segmentPlanes(cloud_after_axis_downsampling, plane_coefficients, 0.05, input_msg, nh);

    
    
    // Assuming clustered_cloud is the result of clusterBasedOnNormals
    // pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud = clusterBasedOnNormals(cloud_after_axis_downsampling, 0.06, 50, 200, distance_threshold, input_msg, nh);

    // Assuming remaining_cloud is the result of the original segmentation
    // pcl::PointCloud<pcl::PointXYZ>::Ptr refined_cloud = refineSegmentation(clustered_cloud, 0.05, input_msg, nh);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud = euclideanClusterAndSegment(cloud_after_axis_downsampling, 0.06, 50, 200, 0.05, input_msg, nh);

// 
    ROS_INFO("-----------------------------------------------");
    
    // Step 3: Calculate and print plane variances
    // for (size_t i = 0; i < plane_coefficients.size(); ++i)
    // {
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr current_plane(new pcl::PointCloud<pcl::PointXYZ>);
    //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //     pcl::ExtractIndices<pcl::PointXYZ> extract;
    //     extract.setInputCloud(remaining_cloud);
    //     // extract.setInputCloud(clustered_cloud);
    //     extract.setIndices(inliers);
    //     extract.setNegative(false);
    //     extract.filter(*current_plane);

    //     double variance = calculatePlaneVariance(current_plane);
    //     ROS_INFO("Variance of plane %zu: %f", i, variance);
    // }

    // ROS_INFO(" ");
    // ROS_INFO("//////////////////////////////////////////////////////////////////////");
    
    // // Introducing a delay for analyzing results
    // ros::Duration(1.0).sleep();
}

// ROS main function
int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_node");
    ros::NodeHandle nh;

    // Publishers
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
    pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
    pub_after_mls = nh.advertise<sensor_msgs::PointCloud2>("/mls_cloud", 1);
    
    pub_after_rotation = nh.advertise<sensor_msgs::PointCloud2>("/rotated_cloud", 1);

    pub_after_plane_segmentation = nh.advertise<sensor_msgs::PointCloud2>("/plane_cloud", 1);

    pub_after_plane_1 = nh.advertise<sensor_msgs::PointCloud2>("/plane_1", 1);
    pub_after_plane_2 = nh.advertise<sensor_msgs::PointCloud2>("/plane_2", 1);
    pub_after_plane_3 = nh.advertise<sensor_msgs::PointCloud2>("/plane_3", 1);
    pub_after_plane_4 = nh.advertise<sensor_msgs::PointCloud2>("/plane_4", 1);
    marker_pub = nh.advertise<visualization_msgs::Marker>("segmented_plane_marker", 1);

    // Subscribing to Lidar Sensor topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));

    ros::spin();

    return 0;
}