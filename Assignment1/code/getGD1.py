@mfunction("result")
def GetGd1(uc = None, gc = None, omega = None):
    result = ((uc - gc) *elmul* omega) *elmul* 2
