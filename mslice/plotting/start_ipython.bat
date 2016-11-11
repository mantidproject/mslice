::set PATH=C:\Users\OMi32458\Workspace\mantid\external\src\ThirdParty\lib\qt4\bin;C:\Users\OMi32458\Workspace\mantid\external\src\ThirdParty\bin;%PATH%
:: fix up pyqt path requisites
:: add embed_in_qt to python_path
<mantid_source_dir>\\mantid\\external\\src\\ThirdParty\\lib\\python2.7\\Scripts\\ipython.cmd --matplotlib -i --c="run %~dp0start_script.ipy"
