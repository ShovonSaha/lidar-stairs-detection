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
#include <pcl/features/normal_3d_omp.h> // Use OpenMP version for faster computation
#include <pcl/common/common_headers.h>
#include <pcl/common/pca.h>
<<<<<<< HEAD
=======
#include <pcl/kdtree/kdtree_flann.h>
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.

#include <visualization_msgs/Marker.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <cmath>
#include <boost/make_shared.hpp> // For creating shared_ptr instances
<<<<<<< HEAD



=======
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.


// ROS Publishers
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_passthrough_z;
<<<<<<< HEAD

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
// std::vector<pcl::ModelCoefficients> plane_coefficients(4);  // Assuming you want to find 4 planes

std::vector<ros::Publisher> plane_publishers; // Publishers for segmented planes
std::vector<pcl::ModelCoefficients> plane_coefficients; // To story.e coefficients of planes






=======
ros::Publisher pub_after_axis_downsampling;




// Function to publish a point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& publisher, const sensor_msgs::PointCloud2ConstPtr& original_msg)
{
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = original_msg->header;
    publisher.publish(output_msg);
}
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.

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


<<<<<<< HEAD
pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterZ(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(-1.0, 0.3);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
  pass.filter(*cloud_filtered_z);

  return cloud_filtered_z;
}


=======
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.
pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplingAlongAxis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& axis, double min_limit, double max_limit)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(0.08, 0.08, 0.08);  // Set an initial leaf size
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

void performEuclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, ros::NodeHandle& nh) {
    // Create a KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
<<<<<<< HEAD
    pcl::PointCloud<pcl::PointNormal> mls_points;
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

    mls.setComputeNormals(false); // We don't need normals in this case
    mls.setInputCloud(cloud);
    mls.setPolynomialOrder(2);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(0.1); // Adjust based on your dataset

    mls.process(mls_points);
=======
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.1); // 0.1 = 10cm
    ec.setMinClusterSize(10); // Minimum size of a cluster
    ec.setMaxClusterSize(500); // Maximum size of a cluster
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    ROS_INFO("Number of clusters found: %d", static_cast<int>(cluster_indices.size()));

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : it->indices)
            cloud_cluster->points.push_back(cloud->points[idx]); //*

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        ROS_INFO("Publishing cluster %d with %ld points", j, cloud_cluster->points.size());

        // Publish each cluster
        std::string topic_name = "/cluster_" + std::to_string(j);
        ros::Publisher pub_cluster = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 1, true);
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_cluster, output);
        output.header.frame_id = "map"; // Set to your global frame
        output.header.stamp = ros::Time::now();
        pub_cluster.publish(output);
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.

        ++j;
    }
}


<<<<<<< HEAD

=======
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.






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


<<<<<<< HEAD
// Function to publish a segmented plane as a marker
void publishSegmentedPlaneMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& segmented_plane, const ros::Publisher& marker_publisher, const pcl::ModelCoefficients::Ptr& coefficients)
{
    // Create a marker for the segmented plane
    visualization_msgs::Marker plane_marker;
    plane_marker.header.frame_id = segmented_plane->header.frame_id;  // Assuming the cloud's frame is relevant
    plane_marker.header.stamp = ros::Time::now();
    plane_marker.ns = "segmented_plane";
    plane_marker.id = marker_publisher.getNumSubscribers();  // Use the number of subscribers as an ID
    plane_marker.type = visualization_msgs::Marker::CUBE;  // Use CUBE for a marker representing planes
    plane_marker.action = visualization_msgs::Marker::ADD;

    // Calculate the centroid of the segmented plane
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*segmented_plane, centroid);
    geometry_msgs::Point centroid_point;
    centroid_point.x = centroid[0];
    centroid_point.y = centroid[1];
    centroid_point.z = centroid[2];

    // Set the marker properties
    plane_marker.pose.position = centroid_point;  // Set the position to the centroid
    plane_marker.scale.x = 0.2;
    plane_marker.scale.y = 0.7;
    plane_marker.scale.z = 0.02;
    plane_marker.color.a = 0.4;   // Opacity
    plane_marker.color.r = 1.0;
    plane_marker.color.g = 0.0;
    plane_marker.color.b = 0.0;

    // Calculate the orientation based on the plane's normal vector
    Eigen::Vector3d normal_vector(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    normal_vector.normalize();  // Ensure the normal vector is normalized

    // // Create a quaternion from the axis-angle representation
    // Eigen::AngleAxisd rotation(normal_vector.cross(Eigen::Vector3d::UnitZ()), std::acos(normal_vector.dot(Eigen::Vector3d::UnitZ())));
    // Eigen::Quaterniond quat(rotation);
    
    // Create a quaternion from the axis-angle representation
    Eigen::Quaterniond quat(Eigen::AngleAxisd(std::acos(normal_vector.dot(Eigen::Vector3d::UnitZ())), normal_vector.cross(Eigen::Vector3d::UnitZ())));


    // Set the orientation using the quaternion
    plane_marker.pose.orientation.x = quat.x();
    plane_marker.pose.orientation.y = quat.y();
    plane_marker.pose.orientation.z = quat.z();
    plane_marker.pose.orientation.w = quat.w();

    // Publish the marker
    marker_publisher.publish(plane_marker);

    // Print the equation of the plane
    ROS_INFO("Equation of Plane %u: Ax + By + Cz + D = 0", marker_publisher.getNumSubscribers());
    ROS_INFO("A: %f, B: %f, C: %f, D: %f", coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
}





void processClustersAndPublish(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, ros::NodeHandle& nh, const sensor_msgs::PointCloud2ConstPtr& original_msg) {
    // Step 1: Cluster the nearby points
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.06); // 6cm tolerance
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);

    // Iterate over each cluster
    int cluster_id = 0;
    for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it, ++cluster_id) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : it->indices) {
            cluster_cloud->points.push_back(input_cloud->points[idx]);
        }

        // Immediately publish the cluster before fitting a plane
        auto cluster_pub = getPlanePublisher(cluster_id, nh);
        publishProcessedCloud(cluster_cloud, cluster_pub, original_msg);
=======
void visualizeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    pcl::visualization::PCLVisualizer viewer("Normals Visualization");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Dark background for better visibility
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");

    // Add normals to the viewer with a specific scale factor for better visibility
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, 0.05, "normals");

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.

        // Step 2: Try to fit a best plane in each cluster
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cluster_cloud);
        seg.segment(*inliers, *coefficients);

<<<<<<< HEAD
        if (inliers->indices.empty()) {
            ROS_WARN("Could not estimate a planar model for the given dataset for cluster %d.", cluster_id);
            continue;
        }

        // Step 3: If a plane is successfully fitted, publish the plane as a marker
        auto marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker_" + std::to_string(cluster_id), 1, true);
        publishSegmentedPlaneMarker(cluster_cloud, marker_pub, coefficients);
    }
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



void estimateNormalsAdaptivePCA(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    pcl::PointCloud<pcl::Normal>::Ptr& normals,
    int minPoints = 20) // Minimum number of points for stable estimation
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normals->resize(cloud->size());

    for (size_t i = 0; i < cloud->size(); ++i) {
        std::vector<int> indices;
        std::vector<float> sqrDistances;
        
        // Dynamically adjust the search radius based on local point density
        tree->nearestKSearch(cloud->points[i], minPoints, indices, sqrDistances);
        
        // Extract the neighborhood
        pcl::PointCloud<pcl::PointXYZ>::Ptr neighborhood(new pcl::PointCloud<pcl::PointXYZ>);
        for (int idx : indices) {
            neighborhood->points.push_back(cloud->points[idx]);
        }

        // Apply PCA on the neighborhood
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(neighborhood);
        Eigen::Vector3f normal = pca.getEigenVectors().col(2); // The smallest eigenvector

        // Ensure normal is outward by comparing with vector towards sensor origin
        Eigen::Vector4f point = (*cloud)[i].getVector4fMap();
        if (normal.dot(point.head<3>()) > 0) {
            normal = -normal;
        }

        normals->points[i].normal_x = normal(0);
        normals->points[i].normal_y = normal(1);
        normals->points[i].normal_z = normal(2);
    }
}


void visualizeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    pcl::visualization::PCLVisualizer viewer("Normals Visualization");
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Dark background for better visibility
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");

    // Add normals to the viewer with a specific scale factor for better visibility
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, 0.05, "normals");

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}







// std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int maxIterations, int minPoints, double distanceThreshold) {
//     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentedPlanes;

//     pcl::PointCloud<pcl::PointXYZ>::Ptr remainingCloud(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
//     pcl::SACSegmentation<pcl::PointXYZ> seg;
//     seg.setOptimizeCoefficients(true);
//     seg.setModelType(pcl::SACMODEL_PLANE);
//     seg.setMethodType(pcl::SAC_RANSAC);
//     seg.setDistanceThreshold(distanceThreshold);

//     pcl::ExtractIndices<pcl::PointXYZ> extract;
//     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//     pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

//     for (int i = 0; i < maxIterations; ++i) {
//         seg.setInputCloud(remainingCloud);
//         seg.segment(*inliers, *coefficients);
        
//         if (inliers->indices.size() < static_cast<size_t>(minPoints)) {
//             break; // No significant plane found
//         }

//         // Extract the segmented plane from the cloud
//         pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
//         extract.setInputCloud(remainingCloud);
//         extract.setIndices(inliers);
//         extract.setNegative(false);
//         extract.filter(*plane);
//         segmentedPlanes.push_back(plane);

//         // Remove the extracted plane from the remaining cloud
//         extract.setNegative(true);
//         pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>());
//         extract.filter(*tempCloud);
//         remainingCloud = tempCloud; // Direct assignment to swap
//     }

//     return segmentedPlanes;
// }




// // Function to classify planes as steps or risers based on their normal
// void classifyPlanes(const std::vector<pcl::ModelCoefficients::Ptr>& coefficients) {
//     for (const auto& coeff : coefficients) {
//         Eigen::Vector3f normal(coeff->values[0], coeff->values[1], coeff->values[2]);
//         Eigen::Vector3f upVector(0, 0, 1);

//         float angle = acos(normal.dot(upVector));
//         if (std::fabs(angle) < M_PI / 6) { // Threshold angle, e.g., M_PI / 6 = 30 degrees
//             std::cout << "Plane classified as step" << std::endl;
//         } else {
//             std::cout << "Plane classified as riser" << std::endl;
//         }
//     }
// }





// Initialize publishers once, e.g., in your class constructor or setup function
void initPublishers(ros::NodeHandle& nh, size_t num_planes) {
    plane_publishers.clear();
    for (size_t i = 0; i < num_planes; ++i) {
        std::stringstream ss;
        ss << "segmented_plane_" << i;
        plane_publishers.push_back(nh.advertise<sensor_msgs::PointCloud2>(ss.str(), 1, true));
    }
}

// void publishSegmentedPlanes(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& planes) {
//     for (size_t i = 0; i < planes.size() && i < plane_publishers.size(); ++i) {
//         sensor_msgs::PointCloud2 output;
//         pcl::toROSMsg(*planes[i], output);
//         output.header.frame_id = "map"; // Ensure this matches your fixed_frame in RViz
//         output.header.stamp = ros::Time::now();

//         if (i >= plane_publishers.size()) {
//             ROS_WARN("Not enough publishers initialized for the number of planes.");
//             break;
//         }

//         plane_publishers[i].publish(output);
//     }
// }



=======
int getNumberOfPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    return cloud->size();
}




>>>>>>> Archived unused codes and streamlines plane_prob.cpp.

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
<<<<<<< HEAD
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_z = passthroughFilterZ(cloud_after_passthrough_y);
    publishProcessedCloud(cloud_after_passthrough_z, pub_after_passthrough_z, input_msg);
    
    
=======
          
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.
    // Downsampling along X-axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(cloud_after_passthrough_z, "x", 0.0, 2.5);
    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, input_msg);

<<<<<<< HEAD
    // Moving Least Squares (MLS) Surface Smoothing and Reconstruction
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_smoothing = applyMLSSmoothing(cloud_after_passthrough_z);
    publishProcessedCloud(cloud_after_smoothing, pub_after_mls, input_msg);

    // Segmenting planes
    // int maxIterations = 10; // Customize based on your needs
    // int minPoints = 100; // Minimum number of points to consider a plane
    // double distanceThreshold = 0.03; // Adjust based on your LiDAR's resolution and accuracy
    // auto segmentedPlanes = segmentPlanes(cloud_after_passthrough_z, maxIterations, minPoints, distanceThreshold);

    // // Publishing the segmented planes
    // publishSegmentedPlanes(segmentedPlanes);

    // processClustersAndPublish(cloud_after_passthrough_z, nh, input_msg);



    // Normal Estimation
    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_downsampled = computeNormals(cloud_after_axis_downsampling);

    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_passthrough_z);
    // visualizeNormals(cloud_after_passthrough_z, cloud_normals);

    // pcl::PointCloud<pcl::Normal>::Ptr oriented_normals = estimateAndOrientNormals(cloud_after_passthrough_z);
    // visualizeNormals(cloud_after_passthrough_z, oriented_normals);

    // Placeholder for the estimated normals
    // pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // estimateNormalsAdaptivePCA(cloud_after_passthrough_z, normals, 20); // Adjust the minPoints parameter as needed
    // visualizeNormals(cloud_after_passthrough_z, normals);


    // Conditional Euclidean Clustering
    // auto clusters = conditionalEuclideanClustering(cloud_after_passthrough_z, cloud_normals);
    // // Publish clusters
    // publishClusters(cloud_after_passthrough_z, clusters, nh);

    // Rotation based on tilt angle
    float tilt_angle = -15.0; // Example tilt angle in degrees
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(tilt_angle * M_PI / 180.0, Eigen::Vector3f::UnitY()));
    pcl::transformPointCloud(*cloud_after_axis_downsampling, *cloud_rotated, transform);

    // publishProcessedCloud(cloud_rotated, pub_after_rotation, input_msg);

    // // Normal Estimation for rotated pointcloud
    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_after_rotation = computeNormals(cloud_rotated);
    
    // Update the number of planes based on the size of plane_coefficients
    // size_t num_planes = plane_coefficients.size();
    
    // Segment planes
=======
    ROS_INFO("Number of points in the downsampled cloud: %d", getNumberOfPoints(cloud_after_axis_downsampling));

    // Perform Euclidean clustering on the downsampled cloud
    performEuclideanClustering(cloud_after_axis_downsampling, nh);
    
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.
    
    // Segmenting planes
    // int maxIterations = 10; // Customize based on your needs
    // int minPoints = 100; // Minimum number of points to consider a plane
    // double distanceThreshold = 0.03; // Adjust based on your LiDAR's resolution and accuracy
    // auto segmentedPlanes = segmentPlanes(cloud_after_passthrough_z, maxIterations, minPoints, distanceThreshold);

<<<<<<< HEAD
    // segmentAndPublishPlanes(cloud_after_axis_downsampling, nh, input_msg);

    // Assuming you want to find 4 planes
    // size_t num_planes = 4;
    // double distance_threshold = 0.05;
    
    // parameters passed: segmentPlanes(input pointcloud, plane_coefficients, distance threshold, sensor msg, node handle)
    // pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud = segmentPlanes(cloud_after_axis_downsampling, plane_coefficients, 0.05, input_msg, nh);
=======
    // // Publishing the segmented planes
    // publishSegmentedPlanes(segmentedPlanes);
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.

    // Normal Estimation
    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_downsampled = computeNormals(cloud_after_axis_downsampling);

    // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud_after_passthrough_z);
    // visualizeNormals(cloud_after_passthrough_z, cloud_normals);

    
    // // Introducing a delay for analyzing results
    // ros::Duration(1.0).sleep();


    // ROS_INFO("----------------------------------------------------------------");
}

// ROS main function
int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_node");
    ros::NodeHandle nh;



    // Publishers
    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
<<<<<<< HEAD
    pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);
=======
    // pub_after_passthrough_z = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_z", 1);
    // pub_after_passthrough_x = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_x", 1);
>>>>>>> Archived unused codes and streamlines plane_prob.cpp.

    pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);
    
<<<<<<< HEAD
    pub_after_rotation = nh.advertise<sensor_msgs::PointCloud2>("/rotated_cloud", 1);

    // pub_after_plane_segmentation = nh.advertise<sensor_msgs::PointCloud2>("/plane_cloud", 1);

    // pub_after_plane_1 = nh.advertise<sensor_msgs::PointCloud2>("/plane_1", 1);
    // pub_after_plane_2 = nh.advertise<sensor_msgs::PointCloud2>("/plane_2", 1);
    // pub_after_plane_3 = nh.advertise<sensor_msgs::PointCloud2>("/plane_3", 1);
    // pub_after_plane_4 = nh.advertise<sensor_msgs::PointCloud2>("/plane_4", 1);
    marker_pub = nh.advertise<visualization_msgs::Marker>("segmented_plane_marker", 1);
    
    
    // Assuming initPublishers() and publishSegmentedPlanes() have been adjusted
    // to manage publishers as suggested in previous responses.
    size_t num_expected_planes = 5; // Adjust based on your application's needs
    initPublishers(nh, num_expected_planes);



=======

>>>>>>> Archived unused codes and streamlines plane_prob.cpp.

    // Subscribing to Lidar Sensor topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, boost::bind(pointcloud_callback, _1, boost::ref(nh)));

    ros::spin();

    return 0;
}