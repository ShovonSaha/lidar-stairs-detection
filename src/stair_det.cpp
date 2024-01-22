#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/organized.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_labeled_clusters.h>
#include <pcl/common/io.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/bilateral.h>
#include <visualization_msgs/Marker.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/normal_3d.h>






// ROS Publishers
ros::Publisher pub_after_passthrough_y;
ros::Publisher pub_after_axis_downsampling;

// ros::Publisher pub_after_plane_segmentation;
ros::Publisher pub_after_plane_1;
ros::Publisher pub_after_plane_2;
ros::Publisher pub_after_plane_3;
ros::Publisher pub_after_plane_4;

ros::Publisher marker_pub;




// Function to publish a segmented plane as a marker
void publishSegmentedPlaneMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& segmented_plane, const ros::Publisher& marker_publisher, const sensor_msgs::PointCloud2ConstPtr& original_cloud_msg)
{
    // Create a marker for the segmented plane
    visualization_msgs::Marker plane_marker;
    plane_marker.header.frame_id = segmented_plane->header.frame_id;  // Assuming the cloud's frame is relevant
    plane_marker.header.stamp = ros::Time::now();
    plane_marker.ns = "segmented_plane";
    plane_marker.id = 0;
    plane_marker.type = visualization_msgs::Marker::POINTS;  // Use POINTS for a marker representing points
    plane_marker.action = visualization_msgs::Marker::ADD;

    // Set the marker properties
    plane_marker.points.resize(segmented_plane->size());
    for (size_t i = 0; i < segmented_plane->size(); ++i)
    {
        // Convert each point in the segmented plane to a geometry_msgs::Point
        geometry_msgs::Point point;
        point.x = segmented_plane->points[i].x;
        point.y = segmented_plane->points[i].y;
        point.z = segmented_plane->points[i].z;
        plane_marker.points[i] = point;
    }

    plane_marker.scale.x = 0.05;  // Adjust the scale as needed
    plane_marker.scale.y = 0.05;
    plane_marker.color.a = 1.0;  // Opacity
    plane_marker.color.r = 1.0;
    plane_marker.color.g = 0.0;
    plane_marker.color.b = 0.0;

    // Publish the marker
    marker_publisher.publish(plane_marker);
}



// Function to publish a segmented plane as a marker with a unique ID
void publishSegmentedPlaneMarkerID(const SegmentedPlane& segmented_plane,
                                  const ros::Publisher& marker_publisher,
                                  const sensor_msgs::PointCloud2ConstPtr& original_cloud_msg,
                                  int marker_id)
{
    // Create a marker for the segmented plane with a unique ID
    visualization_msgs::Marker plane_marker;
    plane_marker.header.frame_id = segmented_plane.cloud->header.frame_id;
    plane_marker.header.stamp = ros::Time::now();
    plane_marker.ns = "segmented_plane";
    plane_marker.id = marker_id;
    plane_marker.type = visualization_msgs::Marker::POINTS;
    plane_marker.action = visualization_msgs::Marker::ADD;

    // Set the marker properties
    plane_marker.points.resize(segmented_plane.cloud->size());
    for (size_t i = 0; i < segmented_plane.cloud->size(); ++i)
    {
        // Convert each point in the segmented plane to a geometry_msgs::Point
        geometry_msgs::Point point;
        point.x = segmented_plane.cloud->points[i].x;
        point.y = segmented_plane.cloud->points[i].y;
        point.z = segmented_plane.cloud->points[i].z;
        plane_marker.points[i] = point;
    }

    plane_marker.scale.x = 0.05;  // Adjust the scale as needed
    plane_marker.scale.y = 0.05;
    plane_marker.color.a = 1.0;
    plane_marker.color.r = 1.0;
    plane_marker.color.g = 0.0;
    plane_marker.color.b = 0.0;

    // Publish the marker
    marker_publisher.publish(plane_marker);
}


// Function to publish a processed point cloud
void publishProcessedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const ros::Publisher& cloud_publisher, const sensor_msgs::PointCloud2ConstPtr& original_cloud_msg)
{
    // Create a new sensor_msgs::PointCloud2 message for the processed point cloud
    sensor_msgs::PointCloud2 processed_cloud_msg;
    pcl::toROSMsg(*cloud, processed_cloud_msg);
    processed_cloud_msg.header = original_cloud_msg->header;

    // Publish the processed point cloud
    cloud_publisher.publish(processed_cloud_msg);
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



// Plane Segmentation


// pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//     pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);

//     pcl::SACSegmentation<pcl::PointXYZ> seg;
//     seg.setOptimizeCoefficients(true);
//     seg.setModelType(pcl::SACMODEL_PLANE);
//     seg.setMethodType(pcl::SAC_RANSAC);
//     seg.setDistanceThreshold(0.1);  // Adjust this threshold according to your data
//     seg.setInputCloud(cloud);
//     seg.segment(*inliers, *plane_coefficients);

//     // Extract the inliers
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::ExtractIndices<pcl::PointXYZ> extract;
//     extract.setInputCloud(cloud);
//     extract.setIndices(inliers);
//     extract.setNegative(false);
//     extract.filter(*cloud_plane);

//     // // Log the number of points in the segmented plane
//     // ROS_INFO("Number of points in segmented plane: %lu", cloud_plane->size());
    
//     return cloud_plane;
// }




// Plane segmentation and Plane Visualization on RViz

// pcl::PointCloud<pcl::PointXYZ>::Ptr segmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
// {
//     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//     pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);

//     pcl::SACSegmentation<pcl::PointXYZ> seg;
//     seg.setOptimizeCoefficients(true);
//     seg.setModelType(pcl::SACMODEL_PLANE);
//     seg.setMethodType(pcl::SAC_RANSAC);
//     seg.setDistanceThreshold(0.1);  // Adjust this threshold according to your data
//     seg.setInputCloud(cloud);
//     seg.segment(*inliers, *plane_coefficients);

//     // Print the plane coefficients
//     ROS_INFO("Plane Coefficients: A = %f, B = %f, C = %f, D = %f",
//              plane_coefficients->values[0], plane_coefficients->values[1],
//              plane_coefficients->values[2], plane_coefficients->values[3]);

//     // Extract the inliers
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::ExtractIndices<pcl::PointXYZ> extract;
//     extract.setInputCloud(cloud);
//     extract.setIndices(inliers);
//     extract.setNegative(false);
//     extract.filter(*cloud_plane);


//     // Visualize the plane using a Marker
//     visualization_msgs::Marker plane_marker;
//     plane_marker.header.frame_id = cloud->header.frame_id;  // Assuming the cloud's frame is relevant
//     plane_marker.header.stamp = ros::Time::now();
//     plane_marker.ns = "plane";
//     plane_marker.id = 0;
//     plane_marker.type = visualization_msgs::Marker::CUBE;  // Use SPHERE for a point marker
//     plane_marker.action = visualization_msgs::Marker::ADD;

//     // Calculate the centroid of the segmented plane
//     Eigen::Vector4f centroid;
//     pcl::compute3DCentroid(*cloud_plane, centroid);

//     // Set the position of the plane marker to the centroid
//     plane_marker.pose.position.x = centroid[0];
//     plane_marker.pose.position.y = centroid[1];
//     plane_marker.pose.position.z = centroid[2];
//     plane_marker.pose.orientation.w = 1.0;
//     plane_marker.scale.x = 1.0;  // Adjust the scale as needed
//     plane_marker.scale.y = 1.0;
//     plane_marker.scale.z = 0.01;  // Adjust the thickness of the plane
//     plane_marker.color.a = 0.5;  // Adjust the alpha value for transparency
//     plane_marker.color.r = 1.0;
//     plane_marker.color.g = 0.0;
//     plane_marker.color.b = 0.0;
//     // plane_marker.color.a = 1.0;  // Opacity
//     // plane_marker.lifetime = ros::Duration(10.0);  // Set the lifetime to 60 seconds (adjust as needed)

//     // Publish the marker
//     // ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("/plane_marker", 1, true);
//     marker_pub.publish(plane_marker);

//     // Return the segmented plane
//     return cloud_plane;
// }


struct SegmentedPlane
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::ModelCoefficients::Ptr coefficients;
};

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentAllPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> all_planes;

    // // Approach 1: Increase Distance Threshold using SACSegmentation
    // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);

    // pcl::SACSegmentation<pcl::PointXYZ> seg1;
    // seg1.setOptimizeCoefficients(true);
    // seg1.setModelType(pcl::SACMODEL_PLANE);
    // seg1.setMethodType(pcl::SAC_RANSAC);
    // seg1.setDistanceThreshold(0.1);  // Increase this threshold based on your data
    // seg1.setInputCloud(cloud);
    // seg1.segment(*inliers, *plane_coefficients);

    // if (inliers->indices.size() == 0)
    // {
    //     ROS_WARN("Plane segmentation failed. No inliers found.");
    //     // Add more details or logging here if needed
    // }

    // // ROS_INFO("Plane Coefficients: A = %f, B = %f, C = %f, D = %f",
    // //      plane_coefficients->values[0], plane_coefficients->values[1],
    // //      plane_coefficients->values[2], plane_coefficients->values[3]);



    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane1(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::ExtractIndices<pcl::PointXYZ> extract1;
    // extract1.setInputCloud(cloud);
    // extract1.setIndices(inliers);
    // extract1.setNegative(false);
    // extract1.filter(*cloud_plane1);
    // all_planes.push_back(cloud_plane1);

    // ROS_INFO("First plane seg done");

    // Approach 2: Iterative RANSAC to segment multiple planes
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *remaining_cloud);

    while (remaining_cloud->size() > 0)
    {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);

        pcl::SACSegmentation<pcl::PointXYZ> seg2;
        seg2.setOptimizeCoefficients(true);
        seg2.setModelType(pcl::SACMODEL_PLANE);
        seg2.setMethodType(pcl::SAC_RANSAC);
        seg2.setDistanceThreshold(0.1);  // Adjust this threshold according to your data
        seg2.setInputCloud(remaining_cloud);
        seg2.segment(*inliers, *plane_coefficients);

        if (inliers->indices.size() == 0)
            break;

        SegmentedPlane segmented_plane;
        segmented_plane.cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract2;
        extract2.setInputCloud(remaining_cloud);
        extract2.setIndices(inliers);
        extract2.setNegative(false);
        extract2.filter(*segmented_plane.cloud);
        segmented_plane.coefficients = plane_coefficients;

        all_planes.push_back(segmented_plane);

        extract2.setNegative(true);
        extract2.filter(*remaining_cloud);

        // ROS_INFO("Plane seg repeat");
    }

    // // Estimate Normals
    // pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(cloud);

    // // Approach 3: Region Growing Segmentation
    // pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    // pcl::IndicesPtr indices(new std::vector<int>);
    // pcl::PassThrough<pcl::PointXYZ> pass;
    // pass.setInputCloud(cloud);
    // pass.setFilterFieldName("z");
    // pass.setFilterLimits(0, 1.5);  // Assuming your stairs are within this height range
    // pass.filter(*indices);

    // pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    // reg.setMinClusterSize(100);
    // reg.setMaxClusterSize(1000000);
    // reg.setSearchMethod(tree);
    // reg.setNumberOfNeighbours(30);
    // reg.setInputCloud(cloud);
    // reg.setIndices(indices);
    // reg.setInputNormals(normals);  // Provide normals if available
    // reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);  // Adjust according to your data
    // reg.setCurvatureThreshold(1.0);  // Adjust according to your data

    // std::vector<pcl::PointIndices> clusters;
    // reg.extract(clusters);

    // for (const auto& cluster_indices : clusters)
    // {
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane3(new pcl::PointCloud<pcl::PointXYZ>);
    //     pcl::ExtractIndices<pcl::PointXYZ> extract3;
    //     extract3.setInputCloud(cloud);
    //     extract3.setIndices(boost::make_shared<const pcl::PointIndices>(cluster_indices));
    //     extract3.setNegative(false);
    //     extract3.filter(*cloud_plane3);
    //     all_planes.push_back(cloud_plane3);
    // }

    // // Approach 4: Conditional Euclidean Clustering
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    // kdtree->setInputCloud(cloud);

    // pcl::ConditionalEuclideanClustering<pcl::PointXYZ> cec;
    // cec.setMinClusterSize(100);
    // cec.setMaxClusterSize(1000000);
    // cec.setClusterTolerance(0.1);  // Adjust according to your data
    // cec.setSearchMethod(kdtree);
    // cec.setInputCloud(cloud);

    // std::vector<pcl::PointIndices> cluster_indices;
    // cec.segment(cluster_indices);

    // for (const auto& cluster : cluster_indices)
    // {
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane4(new pcl::PointCloud<pcl::PointXYZ>);
    //     pcl::ExtractIndices<pcl::PointXYZ> extract4;
    //     extract4.setInputCloud(cloud);
    //     extract4.setIndices(boost::make_shared<const pcl::PointIndices>(cluster));
    //     extract4.setNegative(false);
    //     extract4.filter(*cloud_plane4);
    //     all_planes.push_back(cloud_plane4);
    // }

    return all_planes;
}




int getNumberOfPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    return cloud->size();
}











void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    // Start measuring time
    ros::Time start_time = ros::Time::now();

    // Sensor data acquisition
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    ros::Time rawCloud_end_time = ros::Time::now();

    // Output time taken for getting Raw Point Cloud
    ros::Duration rawCloud_time = rawCloud_end_time - start_time;
    // ROS_INFO("Raw Cloud Acquisition time: %f milliseconds", rawCloud_time.toSec() * 1000.0);

    // ROS_INFO("Number of points in the raw cloud: %d", getNumberOfPoints(cloud));



    // Passthrough Filtering with Y-Axis
    // start_time = ros::Time::now();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_passthrough_y = passthroughFilterY(cloud);

    // Output time taken for passthroughFilterY
    // ros::Time passthroughFilterY_end_time = ros::Time::now();
    // ros::Duration passthroughFilterY_time = passthroughFilterY_end_time - start_time;
    // ROS_INFO("passthroughFilterY time: %f milliseconds", passthroughFilterY_time.toSec() * 1000.0);

    // // Get Number of Points
    // ROS_INFO("Number of points in the cloud_after_passthrough_y cloud: %d", getNumberOfPoints(cloud_after_passthrough_y));

    publishProcessedCloud(cloud_after_passthrough_y, pub_after_passthrough_y, msg);




    // Downsampling Along a Specific Axis using Voxel Grid Downsampling
    // start_time = ros::Time::now();

    // Downsampling along X-axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_axis_downsampling = downsamplingAlongAxis(cloud_after_passthrough_y, "x", 0.0, 2.5);

    // Output time taken for Downsampling
    // ros::Time axis_downsampling_end_time = ros::Time::now();
    // ros::Duration axis_downsampling_time = axis_downsampling_end_time - start_time;
    // ROS_INFO("axis_downsampling_time time: %f milliseconds", axis_downsampling_time.toSec() * 1000.0);

    // // Get Number of Points
    // ROS_INFO("Number of points in the cloud_after_axis_downsampling cloud: %d", getNumberOfPoints(cloud_after_axis_downsampling));

    publishProcessedCloud(cloud_after_axis_downsampling, pub_after_axis_downsampling, msg);

    // Print downsampled points to the terminal
    // for (const auto& point : cloud_after_axis_downsampling->points)
    // {
    // std::cout << "X: " << point.x << ", Y: " << point.y << ", Z: " << point.z << std::endl;
    // }



    // Plane Segmentation
    // start_time = ros::Time::now();

    // Segment all planes
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmented_plane = segmentAllPlanes(cloud_after_axis_downsampling);

    // pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_plane = segmentPlane(cloud_after_axis_downsampling, marker_pub);

    // Output time taken for Plane Segmentation with Downsampled Cloud
    // ros::Time plane_seg_end_time = ros::Time::now();
    // ros::Duration plane_seg_time = plane_seg_end_time - start_time;
    // ROS_INFO("plane_seg_time time: %f milliseconds", plane_seg_time.toSec() * 1000.0); 

    // // Log the number of points in the segmented plane
    // ROS_INFO("Number of points in segmented_all_planes: %lu", segmented_planes->size());

    // Log the number of points in the segmented plane (VECTOR Form)
    ROS_INFO("Number of planes in the segmented vector: %lu", segmented_plane.size());


    // // Publish the segmented plane
    // publishProcessedCloud(segmented_plane, pub_after_plane_segmentation, msg);


    // // Loop over all segmented planes
    // for (size_t i = 0; i < segmented_plane.size(); ++i)
    // {
    //     // Publish each segmented plane
    //     publishProcessedCloud(segmented_plane[i], pub_after_plane_segmentation, msg);
    //     // publishSegmentedPlaneMarker(segmented_plane[i], marker_pub, msg);std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmented_plane = segmentAllPlanes(cloud_after_axis_downsampling);


    // Publish each segmented plane and its marker
    for (size_t i = 0; i < segmented_planes.size(); ++i){
        // Publish the segmented plane point cloud
        publishProcessedCloud(segmented_planes[i].cloud, pub_after_plane[i], msg);

        // Publish the segmented plane as a marker with a unique ID
        publishSegmentedPlaneMarker(segmented_planes[i], pub_after_plane_marker, msg, i);
    }


    publishProcessedCloud(segmented_plane[0], pub_after_plane_1, msg);
    publishProcessedCloud(segmented_plane[1], pub_after_plane_2, msg);
    publishProcessedCloud(segmented_plane[2], pub_after_plane_3, msg);
    publishProcessedCloud(segmented_plane[3], pub_after_plane_4, msg);

    // Log the number of points in the segmented plane
    ROS_INFO("Number of points in segmented_plane_1: %lu", segmented_plane[0]->size());
    ROS_INFO("Number of points in segmented_plane_2: %lu", segmented_plane[1]->size());
    ROS_INFO("Number of points in segmented_plane_3: %lu", segmented_plane[2]->size());
    ROS_INFO("Number of points in segmented_plane_4: %lu", segmented_plane[3]->size());


    // // Log the number of points in the segmented plane
    // ROS_INFO("Number of planes found: %lu", segmented_plane->size());


    // Log the number of points in the segmented plane
    // ROS_INFO("Number of points in segmented plane: %d", getNumberOfPoints(cloud_plane));  // Publish the segmented plane
    // publishProcessedCloud(cloud_plane, pub_after_plane_segmentation, msg);
    // segmentPlane(cloud_after_outlier_removal, msg);

    // Visualization of the segmented plane
    // pcl::visualization::PCLVisualizer::Ptr viewer3(new pcl::visualization::PCLVisualizer("Segmented Plane Viewer"));
    // viewer3->addPointCloud<pcl::PointXYZ>(cloud_after_downsampling, "cloud");
    // viewer3->addPlane(*plane_coefficients, "plane");
    // viewer3->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    // viewer3->spin();






    // Creating line separation for ease of reading
    ROS_INFO("------------------------------------------------------------------");

}










int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_node");

    ros::NodeHandle nh;


    // Publishers

    pub_after_passthrough_y = nh.advertise<sensor_msgs::PointCloud2>("/passthrough_cloud_y", 1);
    pub_after_axis_downsampling = nh.advertise<sensor_msgs::PointCloud2>("/axis_downsampled_cloud", 1);

    // pub_after_plane_segmentation = nh.advertise<sensor_msgs::PointCloud2>("/segmented_all_planes", 1);
    pub_after_plane_1 = nh.advertise<sensor_msgs::PointCloud2>("/plane_1", 1);
    pub_after_plane_2 = nh.advertise<sensor_msgs::PointCloud2>("/plane_2", 1);
    pub_after_plane_3 = nh.advertise<sensor_msgs::PointCloud2>("/plane_3", 1);
    pub_after_plane_4 = nh.advertise<sensor_msgs::PointCloud2>("/plane_4", 1);

    marker_pub = nh.advertise<visualization_msgs::Marker>("segmented_plane_marker", 1);

    // Subcribing to Lidar Sensor topic
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/scan_3D", 1, pointcloud_callback);

    ros::spin();

    return 0;
}