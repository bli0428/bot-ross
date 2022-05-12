from wlkata_mirobot import WlkataMirobot
from wlkata_mirobot import WlkataMirobotTool

arm = WlkataMirobot(portname='/dev/cu.usbserial-1450')

arm.home()