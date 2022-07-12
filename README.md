### Reproducing the experimental evaluations

The code in this repository can be used to reproduce the experimental evaluations seen in the paper

> Albin Heimerson, Johan Ruuskanen, Johan Eker. "Automatic Differentiation over Fluid Models for Holistic Load Balancing", to appear at ACSOS 2022

The experiments are run on the [FedApp sandbox](https://github.com/JohanRuuskanen/FedApp), while the models are extracted and evaluated using `julia-1.7.0`.

In order to recreate the figures seen in the paper, you must first deploy the sandbox with the two example application deployments before running the experiments.



#### Deploy the FedApp sandbox

Access to an OpenStack cloud is required to deploy the sandbox out-of-the-box.  The sandbox can possibly also be deployed on other infrastructures,  but this requires some tinkering. 

Follow the steps in shown in the [sanbox repo](https://github.com/JohanRuuskanen/FedApp), and deploy the gateway with 4 vCPU and 16 Gb of RAM, along with 3 cluster each with 4 virtual machines, each with 4 vCPU and 8 Gb of RAM. 

Finally, using TC netem add two Pareto distributed delays between clusters 1 and 2, and clusters 1 and 3. The delays should be the same in both directions, with the values

​		1 <--> 2: 25ms mean, 5ms jitter, 25% correlation,

​		1 <--> 3: 50ms mean, 10ms jitter, 25% correlation.



#### Application deployment

In the paper, two experiments over two different deployments of the application are considered. The first deployment has 2 replicas of the backend service, and can be found in the `application/facedetect` folder. The second has 3 backend replicas and can be found in the  `application/facedetect_3b` folder.

To deploy the example application on the FedApp sandbox, begin by copying  these  folders to the gateway.

We used the [UMass face detection database](<http://vis-www.cs.umass.edu/fddb/> )  to provide the necessary images for loading the application. Download it and extract it to the `application/data` folder on the gateway.

In `application/facedetect{_3b}/apps`, change the gitlab repository in the `build_and_push.sh` script to point to a container registry that you can access.  Then run the following script to build and push all necessary container images.

```
chmod +x build_and_push.sh
./build_and_push.sh
```

Also, change the gitlab repository in each of the service YAMLs

```
backend-v1.yaml
backend-v2.yaml
{backend-v3.yaml for facedetect_3b}
frontend.yaml
storage.yaml
```

The application can then be deploy with

```
python3 deploy.py
```

To later remove it, simply run 

```
python3 remove.py
```

The application deployment with 2 backend replicas can be reached on port 3002 on the gateway, and the deployment with 3 backend replicas on port 3005. To access either, simply perform a port-forward using e.g. ssh `ssh -L 3002:localhost:3002 ubuntu@GATEWAY_IP` and then visit `http://localhost:3002` on your local computer. 



#### Running the experiments

The cost-minimizing algorithm is implemented in `Julia` and  tested with `Julia-1.7.0`. For the plotting, it further requires `Python` with `matplotlib`. To activate the environment and install dependencies, visit the `online_autodiff/` folder, start `Julia` and type

```Julia
] activate .
] instantiate
```

[DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) was used to solve the fluid model, and [Forwardiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for the automatic differentiation.




##### Offline experiment with 2 backend replicas

After the application has been deployed, visit the `application/facedetect/experiment` folder and update  `lg_settings_grid.py` so that the load generator URL points to one of the nodes on cluster 1.

Gathering the gridded data can then be done with

`python3 run_exp.py`

When completed, this data can be used to run the offline experiment. To do so, visit `online_autodiff/` in the home folder, and start two separate instances of `Julia` using e.g. some terminal multiplexer like [Tmux](https://github.com/tmux/tmux).

Activate the environment with `] activate .` in both instances. Then, in one instance start the process for the cost-minimizing algorithm with 

```Julia
include("online_autodiff.jl")
```

When it is done loading (it will print *Waiting for updated values*), start the offline data playback in the other `Julia` instance with 

```Julia
include("offline_data.jl")
```

After some time, the cost minimization will have converged, and the processes can be interrupted. The resulting experiment data can be plotted by running

```Julia
include("plotting.jl")
```

in any of the two instances. 



##### Online experiment with 3 backend replicas

After the application has been deployed, visit `online_autodiff/` in the home folder and start `Julia`. Activate the environment with  `] activate .` and start the process for the cost-minimizing algorithm with

```Julia
include("online_autodiff.jl")
```

When it is done loading (it will print *Waiting for updated values*), it is time to start the experiment itself. 

In a new terminal window, visit the `application/facedetect_3b/experiment` folder and update  `lg_settings.py` so that the load generator URL points to one of the nodes on cluster 1. Then start the experiment with

```Bash
python3 run_exp.py
```

When the experiment has finished, the resulting data can be plotted by running

```Julia
include("plotting3b.jl")
```

in the `Julia` instance.