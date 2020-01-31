HOST=lin@iccluster
SERVER=$(s)
SCALE=$(sc)
NAMESPACE=.iccluster.epfl.ch
IP=${HOST}${SERVER}${NAMESPACE}
DISK=/mlodata1/tlin
FOLDER=choco_decentralized_code
LOCALPORT=${lp}
REMORTPORT=${rp}
USE_CUDA=${cuda}


# disk stuff...
mount_mlodata1_on_iccluster:
	sudo sshfs -o allow_other,transform_symlinks lin@iccluster${s}.iccluster.epfl.ch:/mlodata1/tlin /home/lin/mount_from_mlodata1


# ssh/scp stuff.
ssh_server:
	ssh ${IP}

ssh_tensorboard:
	ssh -L ${lp}:127.0.0.1:${rp} ${IP}

ssh_notebook:
	ssh -L localhost:${lp}:localhost:${rp} ${IP}

cp_all_gpu:
	rsync -av -e ssh --exclude 'data' ../${FOLDER}/* ${IP}:${DISK}/${FOLDER}
	rsync -av -e ssh --exclude 'data' ../${FOLDER}/* ${IP}:/mloraw1/tlin/${FOLDER}


# docker....
docker_run:
	docker-compose -f iccluster/docker-compose.yml pull
	docker-compose -f iccluster/docker-compose.yml up -d

docker_down:
	docker-compose -f iccluster/docker-compose.yml down


# k8s.
list_k8s_content:
	kubectl config get-contexts

backto_ic_k8s:
	kubectl config use-context lin-mlo-context

backto_g_k8s:
	kubectl config use-context gke_localsgd-198510_us-central1-f_mlo

setup:
	# delete existing worker.
	$(MAKE) delete_cluster
	# create cluster.
	-kubectl create -f iccluster/pod-master.yaml
	-kubectl create -f iccluster/pod-controller.yaml
	# -kubectl create -f iccluster/pod-svc-timeseries.yaml
	# -kubectl create -f iccluster/pod-svc-metadata.yaml

scale:
	kubectl scale statefulsets lin-worker --replicas=${SCALE}

configure:
	# configre cluster.
	cd iccluster && python init_kube_mpi.py --use_cuda ${USE_CUDA} && cd ../
	$(MAKE) send
	kubectl cp iccluster/hostfile lin-master:/mlodata1/tlin/${FOLDER}/iccluster/

run_job:
	-${MAKE} delete_job
	kubectl create -f iccluster/pod-job.yaml

delete_job:
	kubectl delete -f iccluster/pod-job.yaml

send:
	rsync -av -e ssh --exclude 'data' ../${FOLDER}/* ${IP}:/mloraw1/tlin/${FOLDER}

login:
	kubectl exec -it lin-master -- bash

delete_cluster:
	-kubectl delete -f iccluster/pod-master.yaml
	-kubectl delete -f iccluster/pod-controller.yaml
	# -kubectl delete -f iccluster/pod-svc-timeseries.yaml
	# -kubectl delete -f iccluster/pod-svc-metadata.yaml

get_pods:
	kubectl get pods -l user=lin
