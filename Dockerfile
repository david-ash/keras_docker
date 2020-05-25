FROM centos
RUN yum install python36 -y
RUN dnf install python3-pip
RUN yum install sudo -y
RUN sudo yum -y install epel-release
RUN yum -y update
RUN sudo yum -y install gcc gcc-c++ python3-devel atlas atlas-devel gcc-gfortran openssl-devel libffi-devel
#RUN pip3 install --upgrade virtualenv
RUN pip3 install tensorflow
RUN pip3 install keras
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install pillow
RUN sudo mkdir /root/py_files

ENTRYPOINT ["python3"]

