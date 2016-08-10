import abc

class WorkspaceProvider:
    @abc.abstractmethod
    def mtd_getObjectNames(self):
        pass
    @abc.abstractmethod
    def DeleteWorkspace(self,ToBeDeleted):
        pass
    @abc.abstractmethod
    def Load(self,Filename,OutputWorkspace):
        pass
    @abc.abstractmethod
    def GroupWorkspaces(self,InputWorkspaces,OutputWorkspace):
        pass
    @abc.abstractmethod
    def RenameWorkspace(self,selected_workspace,newName):
        pass
    @abc.abstractmethod
    def SaveNexus(self,workspace,path):
        pass