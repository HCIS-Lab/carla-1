# Quick Installation Guideline for CARLA
**This is a guildeline with step-by-step instructions for those who want to quickly set up the environment for Carla.**


## Reference Link
[Carla Documentation 0.9.12](https://carla.readthedocs.io/en/0.9.12/)  
[Carla Documentation 0.9.11](https://carla.readthedocs.io/en/0.9.11/)  
[Conda Docs](https://docs.conda.io/projects/conda/en/latest/index.html)  


## Requirements
**System**: Wiondows/Linux  
**GPU**: GPU RAM 6GB ↑  
**Disk space**: 20 GB ↑  
**Python**: 2.7 + 3.7 (a stable version for now Aug2021)  
**Python package**: pygame, numpy (you can use **pip** to install)  
**Virtual environment:**: Anaconda  




## A Step-by-Step Guideline(Linux)
- The following guideline is based on the [**"Quick start package installation"**](https://carla.readthedocs.io/en/0.9.12/start_quickstart/) in the official document. After basic file installation, we would build a virtual environment for a better version management.

1. Go to [Carla **Relaese** page](https://github.com/carla-simulator/carla/releases) and download compressed package with a version you want.
![](https://i.imgur.com/Gpys7ky.png)
2. Extract the compressed files (**CARLA_0.9.XX.tar.gz**) to a preferable folder. For **GUI linux**, you can directly extract files with **"Right Click"** and then **"Extract to..."** without **tar** commands in terminal.

    **Optional: Import Assets**
    a. Diretly put package files (**AdditionalMaps_0.9.XX.tar.gz** or **CARLA_0.9.10.1_RSS.tar.gz**) into "**Import**" folder in the root folder of carla.* (**Do not extract them!!**)
    b. Run the script in the carla root folder
    >**./ImportAssets.sh**
3. Now you should be able to run a calar host simulator directly with the following command in carla root file.
    >**./CarlaUE4.sh**

4. After above steps, we are going to use **conda** to set up a virtual environment for **Python version management** in order to not influence current Python version of your OS.
    a. **Conda setup**:
    The following content is based-on [Installing on Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) in conda offical website.
    1. Download a installer with the lastest version for Linux on [Anaconda Installers](https://www.anaconda.com/products/individual-d#Downloads), e.g. "64-Bit (x86) Installer (544 MB)" (Sep2021). (You can also choose Miniconda.)
    2. Navigate to your download file and run the following command using your exact filename to start install.
        >**bash Anaconda3-2021.05-Linux-x86_64.sh**
    3. After running "**conda init**", we can go to **carla root folder** to build a virtual environment with the command:
        >
        
    b. **Python version control**:
    
### FAQ
