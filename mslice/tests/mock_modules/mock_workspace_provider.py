from mock import Mock
import sys
import types

module_name = 'mslice.models.workspacemanager.workspace_provider'
workspace_provider = types.ModuleType(module_name)
sys.modules[module_name] = workspace_provider

workspace_provider.get_workspace_handle = Mock(name=module_name+'.get_workspace_handle')
workspace_provider.get_workspace_name = Mock(name=module_name+'.get_workspace_name')
workspace_provider.get_workspace_names = Mock(name=module_name+'.get_workspace_names', return_value=[])
workspace_provider.add_workspace = Mock(name=module_name+'.add_workspace')
workspace_provider.remove_workspace = Mock(name=module_name+'.remove_workspace')
