import os
import ctypes

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

try:
    from OpenGL import EGL
    print("EGL imported successfully!")
    print("EGL Library path:", EGL._types._p.PLATFORM.EGL)
except Exception as e:
    print("Failed to import EGL:")
    print(e)
    import traceback
    traceback.print_exc()

# Let's try to manually load libEGL.so.1
try:
    lib = ctypes.CDLL("libEGL.so.1")
    print("Successfully loaded libEGL.so.1 directly using ctypes:", lib)
except Exception as e:
    print("Failed to load libEGL.so.1 using ctypes:")
    print(e)
