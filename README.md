# Trajectory Optimization of a Spacecraft Swarm (TOSS)

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#test">Test</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project
The goal of the project is to enable an evaluation of the effects of using either single or multi-spacecraft architectures when applied to prospective missions near highly irregular bodies with challenging dynamics, such as comets and asteroids. For that reason, the algorithm focus on optimizing trajectories with impulsive maneuvers using a fast polyhedral gravity model, a Dormand-Prince 8(7)-13M adaptive numerical integration scheme and PyGMO for evolutionary optimization. 

The optimization is based on sampling a large population pool where each candidate solution results in a deterministic trajectory that is obtained by numerical integration. The trajectories are then resampled for a desired fixed time-step and later evaluated according to a user-defined fitness function, such as maximising the visited space around the body while avoiding collisions. Unless the algorithm meets a certain stopping criteria, such as collisions, the process continuous by mutating the population according to the best candidate solution and then reiterating through the process again. The optimisation scheme is followed through for each spacecraft at the time, while keeping previous results in memory in order to guide the optimisation of the next spacecraft trajectory. 

<p align="center">
  <embed src="https://github.com/rasmusmarak/TOSS/blob/Adding-Readme-to-repo/docs/source/TOSS_Optimization_Scheme.pdf" type="application/pdf">
  <p align="center">
    Trajectory Optimisation of a Spacecraft Swarm
    <br />
    <a href="https://github.com/rasmusmarak/TOSS/issues">Report Bug</a>
    ·
    <a href="https://github.com/rasmusmarak/TOSS/issues">Request Feature</a>
  </p>
</p>

In particular, TOSS is designed to be:

- **open-source**: the source code of TOSS is publicly available.
- **decentralised**: each function and capability within the module can be adopted to generate trajectories for a wider variety of objectives and celestial bodies.
- **user-expandable**: the user can expand the capabilities of the code using the decentralised structure. 

The paper corresponding to this project can be found at [ArXiv](https://doi.org/10.48550/arXiv.2306.01602). 

### Built With

This project is based on:

- [polyhedral-gravity-model](https://github.com/esa/polyhedral-gravity-model) a fast, parallelised version of the polyhedral gravity model implemented in C++ along with a python interface.
- [DESolver](https://github.com/Microno95/desolver) a python library fro solving Initial Value Problems using numerical integrators. 
- [PyGMo](https://esa.github.io/pygmo2/index.html) a scientific C++ library, with a python wrapper, for massively parallel optimization managed by the Advanced Concepts Team at the European Space Agency

For more details than provided by TOSS on these libraries, please refer to their docs.


<!-- GETTING STARTED -->

## Getting Started

This is a brief guide how to set up TOSS.

### Installation

### Building from source

To use the latest code from this repository make sure you have all the requirements installed and then clone the [GitHub](https://github.com/rasmusmarak/TOSS.git) repository as follows ([Git](https://git-scm.com/) required):.

```
git clone https://github.com/rasmusmarak/TOSS
```

To install TOSS you can use [conda](https://docs.conda.io/en/latest/) as follows:

```
cd TOSS
conda env create -f environment.yml
```

This will create a new conda environment called `toss` and install the required software packages.
To activate the new environment, you can use:

```
conda activate toss
```


### Test

After cloning the repository, developers can check the functionality of TOSS by running the following command in the `TOSS/tests` directory:

```sh
pytest
```

<!-- USAGE EXAMPLES -->

## Usage

### Config

TOSS uses a central config file which is passed through the entire program. The default config parameters can be seen [here](https://github.com/rasmusmarak/TOSS/blob/main/toss/resources/default_cfg.toml). 

**[Coming Soon]** Practical usage of the config files will soon be added here through an explanatory notebook (Issue [#48](https://github.com/rasmusmarak/TOSS/issues/48)).

### Use Case 1: Trajectory Optimization of a Swarm Orbiting 67P/Churyumov-Gerasimenko Maximising Gravitational Signal
In this case, the aim is to compute a set of trajectories corresponding to a spacecraft swarm of n spacecraft, each with m impulsive maneuvers. The simulation is done within the context of a prospective mission around the comet 67P/Churyumov-Gerasimenko and the scientific target of maximising the measured gravitational signal. For details on mission parameters, optimization structure and fitness functions, see the corresponding paper on [ArXiv](https://doi.org/10.48550/arXiv.2306.01602).

To run this test case, install TOSS and setup the environment as presented above, then run the following:
```
python toss.py
```
TOSS will then create four csv files
- **run_time**: which refers to the total simulation time.
- **champion_f**: being the fitness value corresponding to the champion chromosome.
- **champion_x**: being the champion chromosome corresponding to the optimal set of trajectories.
- **fitness_list**: which is a list of champion fitness values correponding to each generation.

### Use Case 2: Trajectory Optimization Around an Arbitrary Clestial Body Maximising Gravitation Signal
**[Coming Soon]** (Issue referenced in [#55](https://github.com/rasmusmarak/TOSS/issues/55))

### Use Case 3: Trajectory Optimization with User-Defined Scientific Objective
**[Coming Soon]** (Issue referenced in [#55](https://github.com/rasmusmarak/TOSS/issues/55))


<!-- ROADMAP -->

## Roadmap

See the ###(https://github.com/rasmusmarak/TOSS/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->

## Contributing

The project is open to community contributions. Feel free to open an [issue](https://github.com/rasmusmarak/TOSS/issues) or write us an email if you would like to discuss a problem or idea first.

If you want to contribute, please

1. Fork the project on [GitHub](https://github.com/rasmusmarak/TOSS).
2. Get the most up-to-date code by following this quick guide for installing TOSS from source:
   1. Get [miniconda](https://docs.conda.io/en/latest/miniconda.html) or similar
   2. Clone the repo
   ```sh
   git clone https://github.com/rasmusmarak/TOSS
   ```
   3. Set up the environment. This creates a conda environment called
      `toss` and installs the required dependencies.
   ```sh
   conda env create -f environment.yml
   conda activate toss
   ```

Once the installation is done, you are ready to contribute.
Please note that Pull-Requests should be created from and into the `main` branch.

3. Create your Feature Branch (`git checkout -b feature/yourFeature`)
4. Commit your Changes (`git commit -m 'Add some yourFeature'`)
5. Push to the Branch (`git push origin feature/yourFeature`)
6. Open a Pull Request on the `main` branch.

and we will have a look at your contribution as soon as we can.

Furthermore, please make sure that your Pull-Request passes all automated tests. Review will only happen after that.
Only Pull-Requests created on the `main` branch with all tests passing will be considered.

<!-- LICENSE -->

## License

**[Coming Soon]** Distributed under the GNU General Public License v3.0. See [LICENSE]() for more information.

<!-- CONTACT -->

## Contact
Trajectory Optimization for a Spacecraft Swarm (TOSS) is a thesis project presented for the degree of Master of Science in mathematics, with specialization in Optimization and Systems theory provided by KTH Royal Institute of Technology. The thesis topic was defined within the scope of $\Phi$[-lab@Sweden](https://www.ai.se/en/data-factory/f-lab-sweden) in the frame of a collaboration between [AI Sweden](https://www.ai.se/en/) and the [European Space Agency](https://www.esa.int/) to explore distributed edge learning for space applications. The project was mainly supervised by Pablo Gómez and Emmanuel Blazquez from the Advanced Concepts Team (ACT) at ESA, and examined by Xiaoming Hu from KTH.  

Created by:

- Rasmus Maråk - `rasmusmarak at gmail.com`

Supervised by:

- Pablo Gómez - `pablo.gomez at esa.int`
- Emmanuel Blazquez - `emmanuel.blazquez at esa.int`

Examined by:
- Xiaoming Hu - `hu at kth.se`

Project Link: [https://github.com/rasmusmarak/TOSS](https://github.com/rasmusmarak/TOSS)