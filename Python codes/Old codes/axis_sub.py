import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x_values = []
y_values = []
z_values = []

def x_callback(data):
    global x_values
    x_values.extend([point[0] for point in pc2.read_points(data, field_names=("x",), skip_nans=True)])
    print("Received x data:", x_values)

def y_callback(data):
    y_values.extend([point[1] for point in pc2.read_points(data, field_names=("y",), skip_nans=True)])

def z_callback(data):
    z_values.extend([point[2] for point in pc2.read_points(data, field_names=("z",), skip_nans=True)])


def update_plot(num, x_data, y_data, z_data, line_x, line_y, line_z):
    line_x.set_data(range(len(x_data)), x_data)
    line_y.set_data(range(len(y_data)), y_data)
    line_z.set_data(range(len(z_data)), z_data)
    return line_x, line_y, line_z

def timer_callback(event):
    global x_values, y_values, z_values
    line_x.set_data(range(len(x_values)), x_values)
    line_y.set_data(range(len(y_values)), y_values)
    line_z.set_data(range(len(z_values)), z_values)
    ani.event_source.start()

if __name__ == '__main__':
    rospy.init_node('plot_lidar_data', anonymous=True)

    # Set up subscribers for x, y, and z values
    rospy.Subscriber("/x_values", PointCloud2, x_callback)
    rospy.Subscriber("/y_values", PointCloud2, y_callback)
    rospy.Subscriber("/z_values", PointCloud2, z_callback)

    fig, ax = plt.subplots()
    line_x, = ax.plot([], label="X Values")
    line_y, = ax.plot([], label="Y Values")
    line_z, = ax.plot([], label="Z Values")
    ax.legend()

    ani = animation.FuncAnimation(fig, update_plot, fargs=(x_values, y_values, z_values, line_x, line_y, line_z), interval=100)

    # Add a timer to update the plot
    rospy.Timer(rospy.Duration(1.0), timer_callback)

    plt.show()

    rospy.spin()
