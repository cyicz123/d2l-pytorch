FROM	pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
RUN	apt update && \
	python -m pip install --default-timeout=100 --no-cache -U d2l==0.17.6 jupyter
# RUN	groupadd -g 1000 user && useradd -u 1000 -g user user
# USER	user
ENTRYPOINT ["tail", "-f", "/dev/null"]
