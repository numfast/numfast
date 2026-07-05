import sys
print('sys.path[0]:', repr(sys.path[0]))
print('cwd:', __import__('os').getcwd())
import numfast
print('numfast module:', numfast.__file__)
print('AgentSDK:', numfast.AgentSDK)
