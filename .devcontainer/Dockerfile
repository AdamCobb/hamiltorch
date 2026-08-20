FROM quay.io/jupyter/pytorch-notebook:cuda12-latest

### Environment variables
ENV GITHUB_CLI_VERSION 2.30.0

###########################
### SYSTEM INSTALLATION ###
###########################
USER root

### System dependencies. Feel free to add packages as necessary.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Basic system usage
        lmodern \
        file \
        curl \
        g++ \
        tmux \
        ###################################################
        ### Add your own system dependencies installed  ###
        ### with `apt-get` as needed below this comment ###
        ### Example (note the backslash after name):    ###
        ### neofetch \                                  ###
        ###################################################
        && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/library-scripts


#########################
### USER INSTALLATION ###
#########################
USER ${NB_USER}

### Anaconda (conda/mamba) packages
RUN mamba install -y -c conda-forge --freeze-installed \
        # Jupyter setup
        jupyter-server-proxy=4.1.0 \
        #######################################################
        ### Add your own conda dependencies installed with  ###
        ### `conda/mamba` as needed below this comment      ###
        ### Example (note the backslash after name):        ###
        ### scikit-learn \                                  ###
        #######################################################
        && \
    mamba clean --all

### PyPI (pip) packages
RUN pip install \ 
        nbgitpuller \
        ################################################
        ### Add your own PyPI dependencies installed ###
        ### with `pip` as needed below this comment  ###
        ### Example (note the backslash after name): ###
        ### scikit-ntk \                             ###
        ################################################
        torchdyn \
        arviz \
        termcolor \
        && \
    jupyter labextension enable nbgitpuller 
    

### GitHub CLI Installation
RUN wget https://github.com/cli/cli/releases/download/v${GITHUB_CLI_VERSION}/gh_${GITHUB_CLI_VERSION}_linux_amd64.tar.gz -O - | \
    tar xvzf - -C /opt/conda/bin gh_${GITHUB_CLI_VERSION}_linux_amd64/bin/gh --strip-components=2

### Prints Jupyter server token when terminal is opened
RUN echo "echo \"Jupyter server token: \$(jupyter server list 2>&1 | grep -oP '(?<=token=)[[:alnum:]]*')\"" > ${HOME}/.get-jupyter-url.sh && \
    echo "sh \${HOME}/.get-jupyter-url.sh" >> ${HOME}/.bashrc