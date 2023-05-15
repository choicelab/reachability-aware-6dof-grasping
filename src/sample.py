import math
import random
import numpy as np
import open3d as o3d
import copy
import fcl


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


class GraspPoseSampler(object):
    def __init__(self, pc, nb_grasps, low=0, high=90, viz=False):
        self.low = low
        self.high = high
        self.nb_grasps = nb_grasps
        self.pcd = o3d.geometry.PointCloud()
        self.pcd_numpy = pc
        self.pcd.points = o3d.utility.Vector3dVector(pc)
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        self.gripper_mesh = o3d.io.read_triangle_mesh(
            "/home/lou00015/ReachabilityAwareGrasping/coppelia/franka_gripper_collision_mesh.stl")
        self._T = np.array([[0.0, -1.0, 0.0, 0.005],
                            [1.0, 0.0, 0.0, -0.006],
                            [0.0, 0.0, 1.0, -0.085],
                            [0, 0, 0, 1]])
        centroid = np.average(pc, axis=0)
        for i in range(len(pc)):
            x, normal = self.pcd.points[i], self.pcd.normals[i]
            outward_dir = x - centroid
            if angle(outward_dir, normal) > np.pi/2:
                self.pcd.normals[i] = -normal
        if viz:
            o3d.visualization.draw_geometries([self.pcd])

    def collision_checking(self, R, t):
        object_mesh, _ = self.pcd.compute_convex_hull()

        object_collision_mesh = fcl.BVHModel()
        object_collision_mesh.beginModel(len(object_mesh.vertices), len(object_mesh.triangles))
        object_collision_mesh.addSubModel(object_mesh.vertices, object_mesh.triangles)
        object_collision_mesh.endModel()

        gripper_mesh = copy.deepcopy(self.gripper_mesh)
        T = np.concatenate((R, t.reshape((3, 1))), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])))

        T = np.matmul(T, self._T)
        gripper_mesh.transform(T)
        gripper_mesh.compute_vertex_normals()

        gripper_collision_mesh = fcl.BVHModel()
        gripper_collision_mesh.beginModel(len(gripper_mesh.vertices), len(gripper_mesh.triangles))
        gripper_collision_mesh.addSubModel(gripper_mesh.vertices, gripper_mesh.triangles)
        gripper_collision_mesh.endModel()

        req = fcl.CollisionRequest(enable_contact=True)
        res = fcl.CollisionResult()
        fcl.collide(fcl.CollisionObject(object_collision_mesh, fcl.Transform()),
                    fcl.CollisionObject(gripper_collision_mesh, fcl.Transform()),
                    req, res)
        # if not res.is_collision:
        # t_frame = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # t_frame.translate(t)
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([self.pcd, gripper_mesh, object_mesh])
        return res.is_collision

    def generate_gripper_pose(self):
        if self.high == 0:
            omega = random.uniform(0, np.pi)
            up_pose = np.array([[np.cos(omega), -np.sin(omega), 0 ],
                                [np.sin(omega), np.cos(omega),  0 ],
                                [0,             0,              1]])
            x_rot = np.array([[1,  0,  0],
                              [0, -1,  0],
                              [0,  0, -1]])
            pose = np.matmul(x_rot, up_pose)
        else:
            gamma = np.random.randint(self.low+1, self.high)
            gamma = float(abs(gamma))
            ap_z = -math.cos(gamma * 0.0174533)
            ap_x = random.uniform(-math.sqrt(1 - ap_z * ap_z), math.sqrt(1 - ap_z * ap_z))
            ap_y = random.choice([math.sqrt(1 - ap_x * ap_x - ap_z * ap_z), -math.sqrt(1 - ap_x * ap_x - ap_z * ap_z)])
            approach = np.array([ap_x, ap_y, ap_z])
            # cal_gamma = angle(approach, neg_z)
            # print(np.true_divide(cal_gamma,0.0175))

            base0 = np.array([ap_y, -ap_x, 0])
            b0_norm = base0 / np.linalg.norm(base0)
            base1 = np.cross(approach, base0)
            b1_norm = base1 / np.linalg.norm(base1)
            theta = random.uniform(0, 2 * np.pi)
            binormal = np.cos(theta) * b0_norm + np.sin(theta) * b1_norm

            axis = np.cross(approach, binormal)
            approach = np.reshape(approach, (3, 1))
            binormal = np.reshape(binormal, (3, 1))
            axis = np.reshape(axis, (3, 1))
            pose = np.concatenate((binormal, axis, approach), axis=1)
            pose[np.isnan(pose)] = 0
        return pose

    def sample(self):
        pose_set = []
        counter = -1
        while len(pose_set) < self.nb_grasps:
            counter += 1
            if counter > 25000 or len(self.pcd.points) < 10:
                print('sample error')
                return None
            point_idx = random.choice(range(len(self.pcd.points)))
            t, normal = self.pcd.points[point_idx], self.pcd.normals[point_idx]
            R = self.generate_gripper_pose()
            draw_gripper(self.pcd_numpy, R, t)
            theta = angle(R[:, 2], -normal)
            if theta < np.pi / 4:
                collision = self.collision_checking(R, t)
                if not collision:
                    pose_set.append(np.concatenate((R, t.reshape((3, 1))), axis=1))
        return pose_set


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.01):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []
        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def draw_gripper(pc, R, t):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    points = [
        [0, -0.042, 0],
        [0, 0.042, 0],
        [0, 0.04, 0],
        [0, -0.04, 0],
        [0, -0.04, 0.04],
        [0, 0.04, 0.04],
        [0, 0, 0],
        [0, 0, -0.1],
    ]
    points = np.transpose(np.matmul(R[:, :3], np.transpose(points))) + t
    lines = [
        [0, 1],
        [2, 5],
        [3, 4],
        [6, 7]
    ]
    colors = [[0, 1, 0] for _ in range(len(lines))]

    line_mesh1 = LineMesh(points, lines, colors, radius=0.002)
    line_mesh1_geoms = line_mesh1.cylinder_segments
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    t_frame = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    t_frame.translate(t)
    # o3d.visualization.draw_geometries([*line_mesh1_geoms, pcd, t_frame])


