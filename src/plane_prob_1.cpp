#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <visualization_msgs/Marker.h>
#include <random> // Add this line for random number generation

// Define global variables
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0, 1.0);
ros::Publisher pub_after_plane;

ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_axis_downsampling;
ros::Publisher pub_cloud_rotated;
ros::Publisher pub_cloud_with_normals;
ros::Publisher pub_clustered_cloud;

ros::Publisher pub_after_plane_marker;

typedef pcl::PointXYZ PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;

// Function to generate random color
std_msgs::ColorRGBA generateRandomColor() {
    std_msgs::ColorRGBA color;
    color.r = dis(gen);
    color.g = dis(gen);
    color.b = dis(gen);
    color.a = 1.0;
    return color;
}

bool customRegionGrowing(const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap(), point_b_normal = point_b.getNormalVector3fMap();
    if (squared_distance < 10000)
    {
        if (std::abs(point_a_normal.dot(point_b_normal)) > std::cos(30.0f / 180.0f * static_cast<float>(M_PI)))
            return true;
    }
    return false;
}

void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& input_msg)
{
    pcl::PointCloud<PointTypeIO>::Ptr cloud(new pcl::PointCloud<PointTypeIO>);
    pcl::fromROSMsg(*input_msg, *cloud);

    // Step 1: Passthrough Filtering with Y-Axis
    pcl::PassThrough<PointTypeIO> pass_y;
    pass_y.setInputCloud(cloud);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(-0.7, 0.7);
    pcl::PointCloud<PointTypeIO>::Ptr cloud_after_passthrough_y(new pcl::PointCloud<PointTypeIO>);
    pass_y.filter(*cloud_after_passthrough_y);
    sensor_msgs::PointCloud2 output_msg_passthrough_y;
    pcl::toROSMsg(*cloud_after_passthrough_y, output_msg_passthrough_y);
    output_msg_passthrough_y.header = input_msg->header;
    pub_after_passthrough_y.publish(output_msg_passthrough_y);

    // Step 2: Downsampling along X-axis
    pcl::VoxelGrid<PointTypeIO> voxel_grid;
    voxel_grid.setInputCloud(cloud_after_passthrough_y);
    voxel_grid.setLeafSize(0.05, 0.05, 0.05); // Adjust leaf size as needed
    pcl::PointCloud<PointTypeIO>::Ptr cloud_after_downsampling(new pcl::PointCloud<PointTypeIO>);
    voxel_grid.filter(*cloud_after_downsampling);
    sensor_msgs::PointCloud2 output_msg_downsampling;
    pcl::toROSMsg(*cloud_after_downsampling, output_msg_downsampling);
    output_msg_downsampling.header = input_msg->header;
    pub_after_axis_downsampling.publish(output_msg_downsampling);

    // Step 3: Rotation based on tilt angle
    float tilt_angle = 10.0; // Example tilt angle in degrees
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(tilt_angle * M_PI / 180.0, Eigen::Vector3f::UnitX()));
    pcl::transformPointCloud(*cloud_after_downsampling, *cloud_rotated, transform);

    // pcl::toROSMsg(*cloud_rotated, cloud_rotated_msg);
    // cloud_rotated_msg.header.frame_id = "your_frame_id"; // Set appropriate frame id
    // pub_cloud_rotated.publish(cloud_rotated_msg);

    // Step 4: Normal estimation after rotation
    pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals(new pcl::PointCloud<PointTypeFull>);
    pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
    pcl::search::KdTree<PointTypeIO>::Ptr tree(new pcl::search::KdTree<PointTypeIO>());
    ne.setInputCloud(cloud_rotated);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03); // Adjust radius as needed
    ne.compute(*cloud_with_normals);

    // Step 5: Conditional Euclidean Clustering using normals as a condition
    pcl::IndicesClustersPtr clusters(new pcl::IndicesClusters);
    pcl::ConditionalEuclideanClustering<PointTypeFull> cec(true);
    cec.setInputCloud(cloud_with_normals);
    cec.setConditionFunction(&customRegionGrowing);
    cec.setClusterTolerance(0.05); // Adjust cluster tolerance as needed
    cec.setMinClusterSize(100);    // Adjust min cluster size as needed
    cec.setMaxClusterSize(10000);  // Adjust max cluster size as needed
    cec.segment(*clusters);

    // // Step 6: Initialize variables for variance calculation and probability estimation
    // std::vector<float> plane_variances;
    // std::vector<float> plane_probabilities;

    // // Step 7: Iterate through clusters to calculate variance and probability for each plane
    // for (size_t i = 0; i < clusters->size(); ++i)
    // {
    //     pcl::PointCloud<PointTypeFull>::Ptr cluster(new pcl::PointCloud<PointTypeFull>);
    //     for (const auto& index : clusters->at(i).indices)
    //     {
    //         cluster->push_back(cloud_with_normalsyea->points[index]);
    //     }
    //     cluster->width = cluster->size();
    //     cluster->height = 1;
    //     cluster->is_dense = true;

    //     // Calculate variance of the plane
    //     float variance = 0.0;
    //     for (const auto& point : cluster->points)
    //     {
    //         variance += std::pow(point.z - cluster->points[0].z, 2); // Assuming z-axis represents height
    //     }
    //     variance /= cluster->size();
    //     plane_variances.push_back(variance);

    //     // Calculate probability assuming normal distribution
    //     float mean = cluster->points[0].z; // Mean height of the plane
    //     float probability = std::exp(-0.5 * std::pow(variance, -2) * std::pow(point.z - mean, 2)) / std::sqrt(2 * M_PI * variance);
    //     plane_probabilities.push_back(probability);

    //     // Publish resulting point clouds
    //     sensor_msgs::PointCloud2 output_msg_cluster;
    //     pcl::toROSMsg(*cluster, output_msg_cluster);
    //     output_msg_cluster.header = input_msg->header;
    //     pub_after_plane.publish(output_msg_cluster);

        

        // Publish plane as a marker
        // visualization_msgs::Marker marker;
        // marker.header = input_msg->header;
        // marker.ns = "planes";
        // marker.id = i;
        // marker.type = visualization_msgs::Marker::PLANE;
        // marker.action = visualization_msgs::Marker::ADD;
        // marker.pose.position.x = clusters->at(i).centroid[0]; // Assuming centroid represents the position of the plane
        // marker.pose.position.y = clusters->at(i).centroid[1];
        // marker.pose.position.z = clusters->at(i).centroid[2];
        // marker.pose.orientation.x = 0.0;
        // marker.pose.orientation.y = 0.0;
        // marker.pose.orientation.z = 0.0;
        // marker.pose.orientation.w = 1.0;
        // marker.scale.x = 1.0;
        // marker.scale.y = 1.0;
        // marker.scale.z = 1.0;

        // // Generate a random color for the plane marker
        // std_msgs::ColorRGBA color = generateRandomColor();
        // marker.color = color;
        
        // // marker.color.a = 0.5; // Semi-transparent
        // // marker.color.r = 1.0;
        // // marker.color.g = 0.0;
        // // marker.color.b = 0.0;

        // // Publish plane as a marker
        // pub_after_plane_marker.publish(marker);
    // }

    // // Step 8: Print probability values versus the position of the plane in x-axis
    // for (size_t i = 0; i < plane_probabilities.size(); ++i)
    // {
    //     ROS_INFO("Plane %zu: Probability = %f, Position (x) = %f", i, plane_probabilities[i], clusters->at(i).centroid[0]);
    // }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointcloud_processing_node");
    ros::NodeHandle nh;

    // Declare ROS publisher
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/axis_passthrough_cloud", 1);
    pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
    pub_cloud_rotated = nh.advertise<sensor_msgs::PointCloud2>("cloud_rotated", 1);
    pub_cloud_with_normals = nh.advertise<sensor_msgs::PointCloud2>("cloud_with_normals", 1);
    pub_clustered_cloud = nh.advertise<sensor_msgs::PointCloud2>("clustered_cloud", 1);

    
    pub_after_plane = nh.advertise<sensor_msgs::PointCloud2>("/plane_cluster", 1);
    pub_after_plane_marker = nh.advertise<visualization_msgs::Marker>("/plane_markers", 1);

    // Subscribe to point cloud topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, pointcloud_callback);

    ros::spin();

    return 0;
}
