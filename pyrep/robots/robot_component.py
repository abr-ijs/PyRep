from typing import List, Tuple
import pyrep
from pyrep.objects.shape import Shape
from pyrep.backend import sim
from pyrep.const import JointType
from pyrep.objects.object import Object
from pyrep.objects.joint import Joint
from pyrep.const import ObjectType, JointMode
import numpy as np


class RobotComponent(Object):
    """Collection of joints representing arms, end effectors, mobile bases, etc.
    """

    def __init__(self, count: int, name: str, joint_names: List[str],
                 base_name: str = None):
        suffix = '' if count == 0 else '#%d' % (count - 1)
        super().__init__(
            name + suffix if base_name is None else base_name + suffix)
        self._num_joints = len(joint_names)

        # Joint handles
        self.joints = [Joint(jname + suffix)
                       for jname in joint_names]

    def copy(self) -> 'RobotComponent':
        """Copy and pastes the arm in the scene.

        The arm is copied together with all its associated calculation
        objects and associated scripts.

        :return: The new pasted arm.
        """
        # Copy whole model
        handle = sim.simCopyPasteObjects([self._handle], 1)[0]
        name = sim.simGetObjectName(handle)
        # Find the number of this arm
        num = name[name.rfind('#') + 1:]
        if len(num) > 0:
            num = int(num) + 1
        else:
            num = 0
        return self.__class__(num)

    def _get_requested_type(self) -> ObjectType:
        """Gets the type of the object.

        :return: Type of the object.
        """
        return ObjectType(sim.simGetObjectType(self.get_handle()))

    def get_joint_count(self) -> int:
        """Gets the number of joints in this component.

        :return: The number of joints.
        """
        return self._num_joints

    def get_joint_types(self) -> List[JointType]:
        """Retrieves the type of the joints in this component.

        :return: A list containing the types of the joints.
        """
        return [j.get_joint_type() for j in self.joints]

    def get_joint_positions(self) -> List[float]:
        """Retrieves the intrinsic position of the joints.

        See :py:meth:`Joint.get_joint_position` for more information.

        :return: A list of intrinsic position of the joints.
        """
        return [j.get_joint_position() for j in self.joints]

    def set_joint_positions(self, positions: List[float],
                            allow_force_mode=True) -> None:
        """Sets the intrinsic position of the joints.

        See :py:meth:`Joint.set_joint_position` for more information.

        :param positions: A list of positions of the joints (angular or linear
            values depending on the joint type).
        :param allow_force_mode: If True, then the position can be set even
            when the joint mode is in Force mode. It will disable dynamics,
            move the joint, and then re-enable dynamics.
        """
        self._assert_len(positions)

        if not allow_force_mode:
            [j.set_joint_position(p, allow_force_mode)
             for j, p in zip(self.joints, positions)]
            return

        is_model = self.is_model()
        if not is_model:
            self.set_model(True)

        prior = sim.simGetModelProperty(self.get_handle())
        p = prior | sim.sim_modelproperty_not_dynamic
        # Disable the dynamics
        sim.simSetModelProperty(self._handle, p)

        [j.set_joint_position(p, allow_force_mode)
         for j, p in zip(self.joints, positions)]
        [j.set_joint_target_position(p) for j, p in zip(self.joints, positions)]
        sim.simExtStep(True)  # Have to step once for changes to take effect

        # Re-enable the dynamics
        sim.simSetModelProperty(self._handle, prior)
        self.set_model(is_model)

    def get_joint_target_positions(self) -> List[float]:
        """Retrieves the target positions of the joints.

        :return: A list of target position of the joints (angular or linear
            values depending on the joint type).
        """
        return [j.get_joint_target_position() for j in self.joints]

    def set_joint_target_positions(self, positions: List[float]) -> None:
        """Sets the target positions of the joints.

        See :py:meth:`Joint.set_joint_target_position` for more information.

        :param positions: List of target position of the joints (angular or
            linear values depending on the joint type).
        """
        self._assert_len(positions)
        [j.set_joint_target_position(p)
         for j, p in zip(self.joints, positions)]

    def get_joint_target_velocities(self) -> List[float]:
        """Retrieves the intrinsic target velocities of the joints.

         :return: List of the target velocity of the joints (linear or angular
            velocity depending on the joint-type).
         """
        return [j.get_joint_target_velocity() for j in self.joints]

    def set_joint_target_velocities(self, velocities: List[float]) -> None:
        """Sets the intrinsic target velocities of the joints.

        :param velocities: List of the target velocity of the joints (linear
            or angular velocities depending on the joint-type).
        """
        self._assert_len(velocities)
        [j.set_joint_target_velocity(v)
         for j, v in zip(self.joints, velocities)]

    def get_joint_forces(self) -> List[float]:
        """Retrieves the forces or torques of the joints.

        See :py:meth:`Joint.get_joint_force` for more information.

        :return: A list of the forces or the torques applied to the joints
            along/about their z-axis.
        """
        return [j.get_joint_force() for j in self.joints]

    def set_joint_forces(self, forces: List[float]) -> None:
        """Sets the maximum force or torque that the joints can exert.

        See :py:meth:`Joint.set_joint_force` for more information.

        :param forces: The maximum force or torque that the joints can exert.
            These cannot be negative values.
        """
        self._assert_len(forces)
        [j.set_joint_force(f) for j, f in zip(self.joints, forces)]

    def get_joint_velocities(self) -> List[float]:
        """Get the current joint velocities.

        :return: List containing the velocities of the joints (linear or
            angular velocities depending on the joint-type).
        """
        return [j.get_joint_velocity() for j in self.joints]

    def get_joint_intervals(self) -> Tuple[List[bool], List[List[float]]]:
        """Retrieves the interval parameters of the joints.

        See :py:meth:`Joint.get_joint_interval` for more information.

        :return: A tuple containing a list of bools indicates whether the joint
            is cyclic (the joint varies between -pi and +pi in a cyclic manner),
            and a 2D list containing the interval of the joints.
        """
        cyclics, intervals = [], []
        for j in self.joints:
            c, i = j.get_joint_interval()
            cyclics.append(c)
            intervals.append(i)
        return cyclics, intervals

    def set_joint_intervals(self, cyclic: List[bool],
                            intervals: List[List[float]]) -> None:
        """Sets the interval parameters of the joints (i.e. range values).

        See :py:meth:`Joint.set_joint_interval` for more information.

        :param cyclic: List of bools indicates whether the joint is cyclic.
            Only revolute joints with a pitch of 0 can be cyclic.
        :param intervals: 2D list containing the intervals of the joints.
        """
        self._assert_len(cyclic)
        self._assert_len(intervals)
        [j.set_joint_interval(c, i) for j, c, i in zip(
            self.joints, cyclic, intervals)]

    def get_joint_upper_velocity_limits(self) -> List[float]:
        """Gets upper velocity limits of the joints.

         :return: List of the upper velocity limits.
         """
        return [j.get_joint_upper_velocity_limit() for j in self.joints]

    def set_control_loop_enabled(self, value: bool) -> None:
        """Sets whether the control loop is enable for all joints.

        :param value: The new value for the control loop state.
        """
        [j.set_control_loop_enabled(value) for j in self.joints]

    def set_motor_locked_at_zero_velocity(self, value: bool) -> None:
        """Sets if motor is locked when target velocity is zero for all joints.

        When enabled in velocity mode and its target velocity is zero, then the
        joint is locked in place.

        :param value: If the motors should be locked at zero velocity.
        """
        [j.set_motor_locked_at_zero_velocity(value) for j in self.joints]

    def set_joint_mode(self, value: JointMode) -> None:
        """Sets the operation mode of the joint group.

        :param value: The new joint mode value.
        """
        [j.set_joint_mode(value) for j in self.joints]

    def get_visuals(self,
                    search_strings: List[str]=['visual',
                                               'visible']) -> List[Object]:
        """Gets a list of the visual elements of this component.

        Can be useful for methods such as domain randomization.
        Should ideally be overridden for each robot.

        :param search_strings: A list of strings to search for in robot
            elements that identify them as visual.
        :return: A list of visual shapes.
        """
        tree = self.get_objects_in_tree(ObjectType.SHAPE, exclude_base=False)
        return [obj for obj in tree if
                search_strings is None or
                any([string in obj.get_name() for string in search_strings])]

    def randomize_property(self,
                           prop: ['color', 'texture'],
                           group_depth: int=None,
                           seed: int=None,
                           groups: List[List[str]]=None,
                           search_strings=None,
                           color_components: List[float]=['ambient_diffuse',
                                                          'specular',
                                                          'emission',
                                                          'auxiliary'],
                           color_rgb_ranges: List[List[float]]=[[0., 1.],
                                                                [0., 1.],
                                                                [0., 1.]],
                           texture_filename: str=None):
        """Randomize robot component property (i.e. color or texture).

        Visual elements that should be grouped together with the same
        color/texture may be specified with the 'groups' argument.

        E.g., to randomize Panda arm colors in 3 groups:
            groups = [['Panda_link0_visual', 'Panda_link1_visual',
                       'Panda_link2_visual'],
                      ['Panda_link3_visual', 'Panda_link4_visual',
                       'Panda_link5_visual'],
                      ['Panda_link6_visual', 'Panda_link7_visual',
                       'Panda_gripper_visual', 'Panda_leftfinger_visible',
                       'Panda_rightfinger_visual']]
            agent.randomize_property('color', 1, groups=groups)

        NOTE: This is currently only effective at depth 1.

        :param prop: The shape property to be randomized (i.e. 'color' or
            'texture').
        :param group_depth: The recursion depth at which to start grouping
            sub-shapes (None for no grouping, 0 for grouping at first depth
            level, etc.)
        :param seed: A random seed to use for numpy's random number generator.
        :param groups: A list of lists of names of visual elements that should
            be grouped together with the same property.
        :param search_strings: A list of strings to search for to match shape
            elements that should be modified (e.g. 'visual' or 'visible').
        :param color_components: A list of color components of the object to be
            randomized.
        :param color_rgb_ranges: A list of ranges within which the RGB color
            values should be uniformly sampled.
        :param texture_filename: A specific texture image file path for shape
            texture modification.
        :param seed: A random seed to use for numpy's random number generator.
        """
        def find_group_seed(groups, group_seeds, name):
            """A convenience function for finding group/seed of element."""
            for i_group, group in enumerate(groups):
                if name in group:
                    return group_seeds[i_group]
            raise(ValueError('No group seed found!'))

        # Set up grouping/seeding
        if group_depth is not None and group_depth <= 0:
            if seed is None:
                seed = np.random.randint(2**32 - 1)
        group_seeds = []
        if groups is not None and seed is not None:
            for _ in groups:
                group_seeds.append(seed)
                seed += 1
        elif groups is not None and seed is None:
            _seed = np.random.randint(2**32 - 1)
            for _ in groups:
                group_seeds.append(_seed)
                _seed += 1

        # Loop through visual elements
        elements = self.get_visuals(search_strings)
        for element in elements:
            # Handle the grouping/seeding
            if group_depth is not None and group_depth > 0:
                try:
                    seed = find_group_seed(groups, group_seeds,
                                           element.get_name())
                except Exception:
                    if seed is None:
                        seed = np.random.randint(2**32 - 1)
                    else:
                        seed += 1
            # Randomize the visual element
            if group_depth is None:
                _group_depth = None
            else:
                _group_depth = group_depth - 1
            if prop == 'color':
                seed = Shape(element.get_handle()).randomize_color(
                    group_depth=_group_depth,
                    seed=seed,
                    search_strings=search_strings,
                    components=color_components,
                    rgb_ranges=color_rgb_ranges)
            elif prop == 'texture':
                seed = Shape(element.get_handle()).randomize_texture(
                    group_depth=_group_depth,
                    seed=seed,
                    search_strings=search_strings,
                    filename=texture_filename)
            else:
                raise(ValueError('Unknown robot component property: {}'
                                 .format(prop)))

    def randomize_color(self,
                        group_depth: int=None,
                        seed: int=None,
                        groups: List[List[str]]=None,
                        search_strings=None,
                        components: List[float]=['ambient_diffuse',
                                                 'specular',
                                                 'emission',
                                                 'auxiliary'],
                        rgb_ranges: List[List[float]]=[[0., 1.],
                                                       [0., 1.],
                                                       [0., 1.]]):
        """Randomize robot component color.

        Wrapper for RobotComponent.randomize_property() method.

        :param group_depth: The recursion depth at which to start grouping
            sub-shapes of compound shapes to retain the same color (None for no
            grouping, 0 for grouping at first depth level, etc.)
        :param seed: A random seed to use for numpy's random number generator.
        :param groups: A list of lists of names of visual elements that should
            be grouped together with the same color.
        :param search_strings: A list of strings to search for to match shape
            elements that should be modified (e.g. 'visual' or 'visible').
        :param components: A list of color components of the object to be
            randomized.
        :param rgb_ranges: A list of ranges within which the RGB color values
            should be uniformly sampled.
        """
        self.randomize_property(prop='color',
                                group_depth=group_depth,
                                seed=seed,
                                groups=groups,
                                search_strings=search_strings,
                                color_components=components,
                                color_rgb_ranges=rgb_ranges)

    def randomize_texture(self,
                          group_depth: int=None,
                          seed: int=None,
                          groups: List[List[str]]=None,
                          search_strings=None,
                          filename: str=None):
        """Randomize robot component texture.

        Wrapper for RobotComponent.randomize_property() method.

        :param group_depth: The recursion depth at which to start grouping
            sub-shapes of compound shapes to retain the same texture (None for
            no grouping, 0 for grouping at first depth level, etc.)
        :param seed: A random seed to use for numpy's random number generator.
        :param groups: A list of lists of names of visual elements that should
            be grouped together with the same texture.
        :param search_strings: A list of strings to search for to match shape
            elements that should be modified (e.g. 'visual' or 'visible').
        :param filename: A specific texture image file path to be used.
        """
        self.randomize_property(prop='texture',
                                group_depth=group_depth,
                                seed=seed,
                                groups=groups,
                                search_strings=search_strings,
                                texture_filename=filename)

    def _assert_len(self, inputs: list) -> None:
        if len(self.joints) != len(inputs):
            raise RuntimeError(
                'Tried to set values for %d joints, but joint group consists '
                'of %d joints.' % (len(inputs), len(self.joints)))
