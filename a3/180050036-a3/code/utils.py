import os
import os.path as osp
import importlib.util
import copy
from dotmap import DotMap

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
	if not osp.isfile(filename):
		raise FileNotFoundError(msg_tmpl.format(filename))

def load_config_file(filepath):
	filename = osp.abspath(osp.expanduser(filepath))
	check_file_exist(filename)
	fileExtname = osp.splitext(filename)[1]
	if fileExtname not in ['.py']:
		raise IOError('Only py type are supported now!')
	"""
	Parsing Config file
	"""
	if filename.endswith('.py'):
		spec = importlib.util.spec_from_file_location("config", filename)
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		configdata = copy.deepcopy(mod.config)
	return DotMap(configdata)