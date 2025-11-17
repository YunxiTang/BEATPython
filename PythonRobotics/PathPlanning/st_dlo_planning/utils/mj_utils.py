import mujoco as mj
import numpy as np
import glfw
from typing import Union, List


########### mujoco wrapped math utils ######
def aa2quat(axis: Union[List, np.ndarray], angle: float):
    """
    convert axis angle into unit quaternion in order of (`qw, qx, qy, qz`)

    Args:

        axis (Union[List, np.ndarray]): `[x, y, z]`

        angle (float): in radian
    """
    if isinstance(axis, List):
        axis_angle = np.array([axis[0], axis[1], axis[2], angle])
    else:
        axis_angle = np.concatenate([axis, angle])

    quat = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mju_axisAngle2Quat(quat, axis_angle[0:3], axis_angle[3])
    return quat.copy()


########### mujoco utils ##########
class InteractiveRender:
    def __init__(
        self,
        model: mj.MjModel,
        width: int = 1200,
        height: int = 900,
        max_geom: int = 5000,
        render_contact: bool = False,
        render_frame: bool = False,
    ):
        # For callback functions
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        # MuJoCo data structures
        self.model = model  # MuJoCo model
        self.cam = mj.MjvCamera()  # Abstract camera
        self.opt = mj.MjvOption()  # visualization options

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(width, height, "Simulator", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)

        if render_contact:
            self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.model.vis.scale.contactwidth = 0.1
            self.model.vis.scale.contactheight = 0.03
            self.model.vis.scale.forcewidth = 0.05
            self.model.vis.map.force = 0.3

        if render_frame:
            self.opt.frame = mj.mjtFrame.mjFRAME_WORLD

        self.scene = mj.MjvScene(self.model, maxgeom=max_geom)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    def mouse_button(self, window, button, act, mods):
        # update button state
        self.button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self.button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self.button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

        # update mouse position
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (
            (not self.button_left)
            and (not self.button_middle)
            and (not self.button_right)
        ):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene, self.cam
        )

    def scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(
            self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam
        )

    def render(self, data: mj.MjData):
        if not glfw.window_should_close(self.window):
            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mj.mjv_updateScene(
                self.model,
                data,
                self.opt,
                None,
                self.cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scene,
            )
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
        else:
            self.close()

    def close(self):
        glfw.terminate()


########### mujoco-developed renderer ##########
"""A utility class to collect render frames from a function that computes a single frame."""
from typing import Any, Callable, List, Optional, Set

# list of modes with which render function returns None
NO_RETURNS_RENDER = {"human"}

# list of modes with which render returns just a single frame of the current state
SINGLE_RENDER = {"single_rgb_array", "single_depth_array", "single_state_pixels"}


class Renderer:
    """This class serves to easily integrate collection of renders for environments that can computes a single render.
    To use this function:
    - instantiate this class with the mode and the function that computes a single frame
    - call render_step method each time the frame should be saved in the list
      (usually at the end of the step and reset methods)
    - call get_renders whenever you want to retrieve renders
      (usually in the render method)
    - call reset to clean the render list
      (usually in the reset method of the environment)
    """

    def __init__(
        self,
        mode: Optional[str],
        render: Callable[[str], Any],
        no_returns_render: Optional[Set[str]] = None,
        single_render: Optional[Set[str]] = None,
    ):
        """Instantiates a Renderer object.
        Args:
            mode (Optional[str]): Way to render
            render (Callable[[str], Any]): Function that receives the mode and computes a single frame
            no_returns_render (Optional[Set[str]]): Set of render modes that don't return any value.
                The default value is the set {"human"}.
            single_render (Optional[Set[str]]): Set of render modes that should return a single frame.
                The default value is the set {"single_rgb_array", "single_depth_array", "single_state_pixels"}.
        """
        if no_returns_render is None:
            no_returns_render = NO_RETURNS_RENDER
        if single_render is None:
            single_render = SINGLE_RENDER

        self.no_returns_render = no_returns_render
        self.single_render = single_render
        self.mode = mode
        self.render = render
        self.render_list = []

    def render_step(self) -> None:
        """
        Computes a frame and save it to the render collection list.
        This method should be usually called inside environment's step and reset method.
        """
        if self.mode is not None and self.mode not in self.single_render:
            render_return = self.render(self.mode)
            if self.mode not in self.no_returns_render:
                self.render_list.append(render_return)

    def get_renders(self) -> Optional[List]:
        """Pops all the frames from the render collection list.
        This method should be usually called in the environment's render method to retrieve the frames collected till this time step.
        """
        if self.mode in self.single_render:
            return self.render(self.mode)
        elif self.mode is not None and self.mode not in self.no_returns_render:
            renders = self.render_list
            self.render_list = []
            return renders

    def reset(self):
        """Resets the render collection list.
        This method should be usually called inside environment's reset method.
        """
        self.render_list = []


# ==============================================
import collections
import os
import time
from threading import Lock

import glfw
import imageio
import mujoco
import numpy as np


def _import_egl(width, height):
    from mujoco.egl import GLContext

    return GLContext(width, height)


def _import_glfw(width, height):
    from mujoco.glfw import GLContext

    return GLContext(width, height)


def _import_osmesa(width, height):
    from mujoco.osmesa import GLContext

    return GLContext(width, height)


_ALL_RENDERERS = collections.OrderedDict(
    [
        ("glfw", _import_glfw),
        ("egl", _import_egl),
        ("osmesa", _import_osmesa),
    ]
)


class RenderContext:
    """Render context superclass for offscreen and window rendering."""

    def __init__(self, model, data, offscreen=True):
        self.model = model
        self.data = data
        self.offscreen = offscreen
        max_geom = 1000

        mujoco.mj_forward(self.model, self.data)

        self.scn = mujoco.MjvScene(self.model, max_geom)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        self._markers = []
        self._overlays = {}

        self._init_camera()
        self._set_mujoco_buffers()

    def _set_mujoco_buffers(self):
        if self.offscreen:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
            if self.con.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
                raise RuntimeError("Offscreen rendering not supported")
        else:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)
            if self.con.currentBuffer != mujoco.mjtFramebuffer.mjFB_WINDOW:
                raise RuntimeError("Window rendering not supported")

    def update_offscreen_size(self, width, height):
        if width != self.con.offWidth or height != self.con.offHeight:
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
            self.con.free()
            self._set_mujoco_buffers()

    def render(self, width, height, camera_id=None, segmentation=False):
        rect = mujoco.MjrRect(left=0, bottom=0, width=width, height=height)

        # # Sometimes buffers are too small.
        # if width > self.con.offWidth or height > self.con.offHeight:
        #     new_width = max(width, self.model.vis.global_.offwidth)
        #     new_height = max(height, self.model.vis.global_.offheight)
        #     self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(rect, self.scn, self.con)

        for gridpos, (text1, text2) in self._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                gridpos,
                rect,
                text1.encode(),
                text2.encode(),
                self.con,
            )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

    def read_pixels(self, width, height, depth=True, segmentation=False):
        rect = mujoco.MjrRect(left=0, bottom=0, width=width, height=height)

        rgb_arr = np.zeros(3 * rect.width * rect.height, dtype=np.uint8)
        depth_arr = np.zeros(rect.width * rect.height, dtype=np.float32)

        mujoco.mjr_readPixels(rgb_arr, depth_arr, rect, self.con)
        rgb_img = rgb_arr.reshape(rect.height, rect.width, 3)

        ret_img = rgb_img
        if segmentation:
            seg_img = (
                rgb_img[:, :, 0]
                + rgb_img[:, :, 1] * (2**8)
                + rgb_img[:, :, 2] * (2**16)
            )
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)

            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            depth_img = depth_arr.reshape(rect.height, rect.width)
            return (ret_img, depth_img)
        else:
            return ret_img

    def _init_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        for i in range(3):
            self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
        self.cam.distance = self.model.stat.extent

    def add_overlay(self, gridpos: int, text1: str, text2: str):
        """Overlays text on the scene."""
        if gridpos not in self._overlays:
            self._overlays[gridpos] = ["", ""]
        self._overlays[gridpos][0] += text1 + "\n"
        self._overlays[gridpos][1] += text2 + "\n"

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

    def close(self):
        """Override close in your rendering subclass to perform any necessary cleanup
        after env.close() is called.
        """
        pass


class RenderContextOffscreen(RenderContext):
    """Offscreen rendering class with opengl context."""

    def __init__(self, width, height, model, data):
        self._get_opengl_backend(width, height)
        self.opengl_context.make_current()
        # self.ctx = mujoco.GLContext(width, height)
        # self.ctx.make_current()

        super().__init__(model, data, offscreen=True)

    def _get_opengl_backend(self, width, height):
        backend = os.environ.get("MUJOCO_GL")
        if backend is not None:
            try:
                self.opengl_context = _ALL_RENDERERS[backend](width, height)
            except KeyError:
                raise RuntimeError(
                    "Environment variable {} must be one of {!r}: got {!r}.".format(
                        "MUJOCO_GL", _ALL_RENDERERS.keys(), backend
                    )
                )

        else:
            for name, _ in _ALL_RENDERERS.items():
                try:
                    self.opengl_context = _ALL_RENDERERS[name](width, height)
                    backend = name
                    break
                except:  # noqa:E722
                    pass
            if backend is None:
                raise RuntimeError(
                    "No OpenGL backend could be imported. Attempting to create a "
                    "rendering context will result in a RuntimeError."
                )


class Viewer(RenderContext):
    """Class for window rendering in all MuJoCo environments."""

    def __init__(self, model, data):
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._transparent = False
        self._contacts = False
        self._render_every_frame = True
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menu = False

        # glfw init
        glfw.init()
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        self.window = glfw.create_window(width // 2, height // 2, "mujoco", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width

        # set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        # get viewport
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        super().__init__(model, data, offscreen=False)

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Pause simulation
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Slows down simulation
        elif key == glfw.KEY_S:
            self._run_speed /= 2.0
        # Speeds up simulation
        elif key == glfw.KEY_F:
            self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        elif key == glfw.KEY_D:
            self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        elif key == glfw.KEY_T:
            img = np.zeros(
                (
                    glfw.get_framebuffer_size(self.window)[1],
                    glfw.get_framebuffer_size(self.window)[0],
                    3,
                ),
                dtype=np.uint8,
            )
            mujoco.mjr_readPixels(img, None, self.viewport, self.con)
            imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = not self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
        # Display coordinate frames
        elif key == glfw.KEY_E:
            self.vopt.frame = 1 - self.vopt.frame
        # Hide overlay menu
        elif key == glfw.KEY_H:
            self._hide_menu = not self._hide_menu
        # Make transparent
        elif key == glfw.KEY_R:
            self._transparent = not self._transparent
            if self._transparent:
                self.model.geom_rgba[:, 3] /= 5.0
            else:
                self.model.geom_rgba[:, 3] *= 5.0
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        # Quit
        if key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            glfw.destroy_window(self.window)
            glfw.terminate()

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self._button_left_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, action, dx / height, dy / height, self.scn, self.cam
            )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_right_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0,
                -0.05 * y_offset,
                self.scn,
                self.cam,
            )

    def _create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        if self._render_every_frame:
            self.add_overlay(topleft, "", "")
        else:
            self.add_overlay(
                topleft,
                "Run speed = %.3f x real time" % self._run_speed,
                "[S]lower, [F]aster",
            )
        self.add_overlay(
            topleft, "Ren[d]er every frame", "On" if self._render_every_frame else "Off"
        )
        self.add_overlay(
            topleft,
            "Switch camera (#cams = %d)" % (self.model.ncam + 1),
            "[Tab] (camera ID = %d)" % self.cam.fixedcamid,
        )
        self.add_overlay(topleft, "[C]ontact forces", "On" if self._contacts else "Off")
        self.add_overlay(topleft, "T[r]ansparent", "On" if self._transparent else "Off")
        if self._paused is not None:
            if not self._paused:
                self.add_overlay(topleft, "Stop", "[Space]")
            else:
                self.add_overlay(topleft, "Start", "[Space]")
                self.add_overlay(
                    topleft, "Advance simulation by one step", "[right arrow]"
                )
        self.add_overlay(
            topleft, "Referenc[e] frames", "On" if self.vopt.frame == 1 else "Off"
        )
        self.add_overlay(topleft, "[H]ide Menu", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            self.add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            self.add_overlay(topleft, "Cap[t]ure frame", "")
        self.add_overlay(topleft, "Toggle geomgroup visibility", "0-4")

        self.add_overlay(bottomleft, "FPS", "%d%s" % (1 / self._time_per_render, ""))
        self.add_overlay(
            bottomleft, "Solver iterations", str(self.data.solver_iter + 1)
        )
        self.add_overlay(
            bottomleft, "Step", str(round(self.data.time / self.model.opt.timestep))
        )
        self.add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)

    def render(self):
        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window
            )
            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    mujoco.MjvPerturb(),
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn,
                )
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.con)
                # overlay items
                if not self._hide_menu:
                    for gridpos, [t1, t2] in self._overlays.items():
                        mujoco.mjr_overlay(
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            gridpos,
                            self.viewport,
                            t1,
                            t2,
                            self.con,
                        )
                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

            # clear overlay
            self._overlays.clear()

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()
