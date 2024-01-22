// ROS
#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>

// PCL
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

class PlaneExtractor
{
public:
    PlaneExtractor(ros::NodeHandle nh) : _nh(nh)
    {
        initialize();
    }

    void initialize()
    {
        // Initialize parameters and publishers
        _pub_inliers = _nh.advertise<sensor_msgs::PointCloud2>("segmented_planes", 2);

        // Set parameters as needed
        _max_distance = 0.005;
        _min_percentage = 5;
        _desired_num_planes = 10;
    }

    void pointCloudCb(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        // Convert to pcl point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_msg(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud_msg);

        // Step 1: Passthrough filter in Y-axis
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passthrough_y = passthroughFilterY(cloud_msg);

        // Step 2: Downsampling along an axis
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled = downsamplingAlongAxis(cloud_passthrough_y, "x", 0.0, 2.5);

        // Loop to find and publish planes
        for (int i = 0; i < _desired_num_planes; ++i)
        {
            // Step 3: Find a plane in the downsampled cloud
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            findPlane(cloud_downsampled, coefficients, inliers);

            // Check result
            if (inliers->indices.size() > 0)
            {
                // Step 4: Publish the segmented plane (inliers)
                publishSegmentedPlane(cloud_downsampled, inliers);

                // Step 5: Remove inliers found from the plane in the current cloud
                removeInliers(cloud_downsampled, inliers);
            }
            else
            {
                ROS_WARN("No plane found in iteration %d", i + 1);
                break;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilterY(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-0.7, 0.7);
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pass.filter(*filtered_cloud);

        return filtered_cloud;
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

    void findPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::ModelCoefficients::Ptr &coefficients, pcl::PointIndices::Ptr &inliers)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(_max_distance);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
    }

    void publishSegmentedPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointIndices::Ptr &inliers)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pub(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (const auto &index : inliers->indices)
        {
            pcl::PointXYZRGB pt;
            pt.x = cloud->points[index].x;
            pt.y = cloud->points[index].y;
            pt.z = cloud->points[index].z;
            pt.r = 255; // You can adjust the color as needed
            pt.g = 0;
            pt.b = 0;
            cloud_pub->points.push_back(pt);
        }

        sensor_msgs::PointCloud2 cloud_publish;
        pcl::toROSMsg(*cloud_pub, cloud_publish);
        cloud_publish.header = msg->header;
        _pub_inliers.publish(cloud_publish);
    }

    void removeInliers(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointIndices::Ptr &inliers)
    {
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZ> cloudF;
        extract.filter(cloudF);
        cloud->swap(cloudF);
    }

private:
    ros::NodeHandle _nh;
    ros::Publisher _pub_inliers;

    double _max_distance;
    double _min_percentage;
    int _desired_num_planes;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "planeExtractor");
    ros::NodeHandle nh("~");

    PlaneExtractor extractor(nh);

    ros::Subscriber sub = nh.subscribe("/your_point_cloud_topic", 1, &PlaneExtractor::pointCloudCb, &extractor);

    ros::spin();

    return 0;
}
