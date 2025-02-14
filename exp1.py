import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
from match_image import *
from PIL import Image
from std_msgs.msg import String


class ImageToCmdVelNode:
    def __init__(self):
        rospy.init_node("image_to_cmd_vel_node")
        self.image_topic = "/camera/rgb/image_raw/compressed"
        self.cmd_vel_topic = "/navigation/cmd_vel"
        self.cmdvelhz = 45
        self.bridge = CvBridge()
        self.latest_image_msg = None  # Store latest received image
        self.current_cmd_list = None
        self.current_cmd_counter = 0
        self.rootdir_topomap = "/root/SelaVPR/exps/room"
        print("Setting up...")
        self.sela = SelaVPRminimal()
        self.reader()
        print("Setup done")

        rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.logger_pub = rospy.Publisher("/explog", String, queue_size=10)
        rospy.Timer(rospy.Duration(1 / self.cmdvelhz), self.publish_cmd_vel)

    @torch.inference_mode()
    def reader(self):
        all_imgs = sorted(os.listdir(os.path.join(self.rootdir_topomap, "images")))
        self.all_imgs_t = [int(a.split(".")[0]) for a in all_imgs]
        all_imgs_paths = [os.path.join(self.rootdir_topomap, "images", a) for a in all_imgs]
        self.all_local_feats, self.all_global_feats, _ = self.sela.run_model(all_imgs_paths)

        self.all_cmdvel = self.read_cmd_vel_file(os.path.join(self.rootdir_topomap, "cmd_vel.txt"))
        self.all_cmdvel_t = list(self.all_cmdvel.keys())

        self.all_localization = self.read_localization_file(os.path.join(self.rootdir_topomap, "localization.txt"))
        self.all_localization_t = list(self.all_localization.keys())

        self.matched_localization_t = list(self.match_closest(self.all_imgs_t, self.all_localization_t))
        self.matched_cmdvel_t = list(self.group_by_range(self.all_imgs_t, self.all_cmdvel_t))

    @staticmethod
    def match_closest(list1, list2):
        list1 = np.array(list1)
        list2 = np.array(list2)
        closest_indices = np.abs(list1[:, None] - list2).argmin(axis=1)
        return list2[closest_indices]

    @staticmethod
    def group_by_range(list1, list2):
        result = {}
        j = 0  # Pointer for list2

        for i in range(len(list1) - 1):
            start, end = list1[i], list1[i + 1]
            result[start] = []

            while j < len(list2) and list2[j] < end:
                if list2[j] >= start:
                    result[start].append(list2[j])
                j += 1

        return result

    def read_cmd_vel_file(self, file_path):
        data_dict = {}
        with open(file_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                values = line.split()
                timestamp = int(values[0])
                data_dict[timestamp] = list(map(float, values[1:]))  # Convert remaining values to float
        return data_dict

    def read_localization_file(self, file_path):
        data_dict = {}
        with open(file_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                values = line.split()
                timestamp = int(values[0])
                data_dict[timestamp] = list(map(float, values[1:]))  # Convert x, y, theta to floats
        return data_dict

    def image_callback(self, msg):
        self.latest_image_msg = msg

    @torch.inference_mode()
    def publish_cmd_vel(self, event):
        if self.latest_image_msg is not None:
            if self.current_cmd_list is None:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(self.latest_image_msg, desired_encoding="bgr8")
                cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_image_rgb)
                query_local_feat, query_global_feat, _ = self.sela.run_model([pil_img])
                K = 5 if len(self.all_imgs_t) > 5 else len(self.all_imgs_t)
                _, indices = self.sela.match_global_feature(query_global_feat, self.all_global_feats, K=K)
                new_indices, hscores = self.sela.match_local_feature(query_local_feat, self.all_local_feats, indices)
                new_indices = new_indices.squeeze().tolist()
                chosen_img_node_index = new_indices[0]
                smsg = String()
                smsg.data = f"HScores: {hscores[0]}, Chosen Image: {chosen_img_node_index}"
                self.logger_pub.publish(smsg)
                chosen_img_node_t = self.all_imgs_t[chosen_img_node_index]
                self.current_cmd_list = self.matched_cmdvel_t[chosen_img_node_t]
                self.current_cmd_counter = 0
                if len(self.current_cmd_list) == 0:
                    self.current_cmd_list = None
                    return
            twist_msg = Twist()
            twist_msg.linear.x = self.current_cmd_list[self.current_cmd_counter][0]
            twist_msg.linear.y = self.current_cmd_list[self.current_cmd_counter][1]
            twist_msg.linear.z = self.current_cmd_list[self.current_cmd_counter][2]
            twist_msg.angular.x = self.current_cmd_list[self.current_cmd_counter][3]
            twist_msg.angular.y = self.current_cmd_list[self.current_cmd_counter][4]
            twist_msg.angular.z = self.current_cmd_list[self.current_cmd_counter][5]
            self.cmd_vel_pub.publish(twist_msg)
            self.current_cmd_counter += 1
            if self.current_cmd_counter >= len(self.current_cmd_list):
                self.current_cmd_list = None


if __name__ == "__main__":
    node = ImageToCmdVelNode()
    rospy.spin()  # Keep the node running
