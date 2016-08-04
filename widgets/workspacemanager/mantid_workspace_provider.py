from workspace_provider import WorkspaceProvider
from mantid.simpleapi import mtd,Load,DeleteWorkspace,Load,GroupWorkspaces,RenameWorkspace,SaveNexus

class MantidWorkspaceProvider(WorkspaceProvider):
    def getWorkspaceNames(self):
        return mtd.getObjectNames()
    def DeleteWorkspace(self,ToBeDeleted):
        return DeleteWorkspace(ToBeDeleted)
    def Load(self,Filename,OutputWorkspace):
        return Load(Filename=Filename,OutputWorkspace=OutputWorkspace)
    def GroupWorkspaces(self,InputWorkspaces,OutputWorkspace):
        return GroupWorkspaces(InputWorkspaces,OutputWorkspace)
    def RenameWorkspace(self,selected_workspace,newName):
        return RenameWorkspace(selected_workspace,newName)
    def SaveNexus(self,workspace,path):
        SaveNexus(workspace,path)
        