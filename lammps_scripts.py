import os
import random


class TemperatureRange:
    """Simple data class to store temperature range information."""

    T_start: float
    T_end: float
    bias: float

    def __init__(
        self,
        T_start: float = 1e-7,
        T_end: float = 1e-7,
        bias: float = 10.0
    ):
        self.T_start = T_start
        self.T_end = T_end
        self.bias = bias

    def __repr__(self) -> str:
        return f"TempRange: start={self.T_start:.3e}, stop={self.T_end:.3e}, bias={self.bias:.2f}"


class LJSimulation:
    """Defines the initial lammps calculation which is used to generate a disordered network.
    Default values work 100%, anything you change may not.
    """

    n_atoms: int
    atom_sizes: list[float]
    box_dim: list[float]
    temperature_range: TemperatureRange = TemperatureRange()
    n_steps: int

    def __init__(
        self,
        n_atoms: int = 200,
        atom_sizes: list[float] | None = None,
        box_dim: list[float] | None = None,
        temperature_range: TemperatureRange | None = None,
        iso_p: float = 0.2,
        p_damp: float = 0.6,
        Kn: float = 1.0,
        n_steps: int = 30000,
        n_parts: list[int] | None = None,
    ):
        if temperature_range is None:
            temperature_range = TemperatureRange()
        if atom_sizes is None:
            atom_sizes = [1.6, 1.4, 1.2, 1.0]
        if box_dim is None:
            box_dim = [-7.0, 7.0, -7.0, 7.0, -0.1, 0.1]
        self.n_atoms = n_atoms
        self.atom_sizes = atom_sizes
        self.n_atom_types = len(atom_sizes)
        self.box_dim = box_dim
        self.temperature_range = temperature_range
        self.iso_p = iso_p
        self.p_damp = p_damp
        self.Kn = Kn
        self.n_steps = n_steps
        self.n_parts = n_parts

    def write_to_file(self, directory: str | None = None):
        create_atoms_section: str = ""
        group_section: str = ""
        group_size_section: str = ""
        if self.n_parts is None:
            for i in range(self.n_atom_types):
                create_atoms_section += f"create_atoms {i + 1} random ${{npart}} {random.randint(1, 999999)} NULL overlap 0.5 maxtry 100\n"
                group_section += f"group size{i + 1} type {i + 1}\n"
                group_size_section += (
                    f"set group size{i + 1} diameter {self.atom_sizes[i]}\n"
                )
        else:
            for i, n in enumerate(self.n_parts):
                create_atoms_section += f"create_atoms {i + 1} random {n} {random.randint(1, 999999)} NULL overlap 0.5 maxtry 100\n"
                group_section += f"group size{i + 1} type {i + 1}\n"
                group_size_section += (f"set group size{i + 1} diameter {self.atom_sizes[i]}\n")

        script_template: str = f"""
#area fraction
# 0.85 is in the liquid state at T=1
# 0.90 is a crystalline solid)
variable afrac equal 1.00

#number of particles in 20x20 area
variable npart  equal ${{afrac}}*{self.n_atoms // self.n_atom_types}


#we use the LJ potential epsilon as energy scale,
#and sigma as length scale.
units		 lj
dimension    2
atom_style	 sphere 
boundary     p p p
neighbor     0.5   bin
neigh_modify every 1 delay 0 check yes


# create 2D box
region box block {self.box_dim[0]} {self.box_dim[1]} {self.box_dim[2]} {self.box_dim[3]} {self.box_dim[4]} {self.box_dim[5]}
create_box {self.n_atom_types} box


# put z=0 all the time
fix 2d  all enforce2d
comm_modify vel yes


# put the particles randomly into the box
{create_atoms_section}


# create groups named size1 to size4 of 4 types of atoms defined above
{group_section}


# sets the sizes of atoms within the defined groups
{group_size_section}


# define soft granualr potential forces (as done in Reid paper).
pair_style gran/hooke {self.Kn} 0.0 0.0 0.0 0.0 0
pair_coeff * *

# minimize energy first to avoid overlapping particles
minimize 1e-10 1e-10 1000 1000
#fix 10 all box/relax aniso ${{Pres}} vmax 0.001


reset_timestep 0
fix 10 all npt/sphere temp {self.temperature_range.T_start} {self.temperature_range.T_end} {self.temperature_range.bias} iso {self.iso_p} {self.iso_p} {self.p_damp} disc


# output time step, temperature, average kinetic and potential energy
# thermo_style custom step temp ke pe
# thermo		100
dump            1 all atom 50 dump.lammpstrj

#time step of integrator
timestep	0.006

#number of integration steps to run
run		{self.n_steps}

write_data coord.dat
"""

        if directory is None:
            with open("lammps.in", "w", encoding="utf8") as lammps_file:
                lammps_file.write(script_template)
        else:
            path = os.path.join(directory, "lammps.in")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf8") as lammps_file:
                lammps_file.write(script_template)


class CompressionSimulation:
    """Defines the compression simulation which is used to generate data for GNN simulator."""

    network_filename: str
    strain: float
    strain_rate: float
    temperature_range: TemperatureRange
    dump_frequency: int
    disable_nph: bool

    def __init__(
        self,
        strain_direction: str = "x",
        box_size: float = 15.0,
        network_filename: str = "network.lmp",
        remap: str = "x",
        dt: float = 0.01,
        strain: float = 0.03,
        strain_rate: float = 1e-5,
        desired_step_size: float = 0.001,
        overwrite_dump_freq: int | None = None,
        random_seed: bool = False,
        disable_nph: bool = False,
        temperature_range: TemperatureRange | None = None,
        lj_active: bool = False,
        lj_params: dict | None = None,
    ) -> None:
        self.network_filename = network_filename
        self.dt = dt
        self.strain = strain
        self.strain_rate = strain_rate
        self.remap = remap
        self.temperature_range = temperature_range
        self.lj_active = lj_active
        self.lj_params = lj_params
        self.desired_step_size = desired_step_size
        self.strain_direction = strain_direction
        self.fix_axis = "y" if self.strain_direction == "x" else "x"
        self.dump_frequency = int(
            (self.strain / self.strain_rate / self.dt)
            / (box_size * self.strain / self.desired_step_size)
        )
        if overwrite_dump_freq is not None:
            self.dump_frequency = overwrite_dump_freq
        self.random_seed = 904297 if random_seed is False else random.randint(1, 999999)
        self.disable_nph = disable_nph
        if temperature_range is None:
            temperature_range = TemperatureRange()

        if self.disable_nph:
            self.nph_line = f"#fix             6 all nph {self.fix_axis} 0.0 0.0 1000.0 ptemp 0.0001"
        else:
            self.nph_line = (f"fix             6 all nph {self.fix_axis} 0.0 0.0 1000.0 ptemp 0.0001")

    def _recalc_dump_freq(self, box_size):
        self.dump_frequency = int(
            (self.strain / self.strain_rate / self.dt)
            / (box_size * self.strain / self.desired_step_size)
        )

    def write_to_file(self, directory: str | None = None):
        deformation_script_template: str = f"""
variable dt equal {self.dt}
variable strain equal {self.strain}
variable srate equal {self.strain_rate}
variable M equal 1000 # averaging time for properties
variable strainsteps equal ${{strain}}/${{dt}}/${{srate}}


dimension 2
include init.mod
include potential.mod


#change_box all triclinic
comm_modify mode single cutoff 10.0 vel yes
atom_modify sort 0 0


# Compute initial state
fix 3 all box/relax x 0 y 0 # xy 0  #MS comment: relax if the network is periodic
minimize ${{etol}} ${{ftol}} ${{maxiter}} ${{maxeval}}
thermo 100
# thermo_style custom step atoms temp press c_vx c_vy c_vz
unfix 3
fix 2d all enforce2d


# These formulas define the derivatives w.r.t. strain components
# Constants uses $, variables use v_ 
timestep        ${{dt}}
variable nsteps equal round(${{strainsteps}})

#fix             4 all nve
#fix		     6 all press/berendsen y  0.0 0.0 1 x 0.0 0.0 10
#fix             6 all npt temp 0.0001 0.0001 1000.0 y 0.0 0.0 1000.0
fix              5 all langevin {self.temperature_range.T_start} {self.temperature_range.T_end} {self.temperature_range.bias} {self.random_seed}
{self.nph_line}
fix     	     9 all deform 1 {self.strain_direction} erate -${{srate}} units box remap {self.remap}

# For PLUMED2
# fix 22 all plumed plumedfile plumed.dat outfile plumed.out

#To avoid flying ice cube phenomenon:
fix 21 all momentum 100 linear 1 1 1 angular

dump 1 all custom {self.dump_frequency} dump.lammpstrj id x y z vx vy vz fx fy fz
dump_modify 1 format line "%d %.15g %.15g %.15g %.8e %.8e %.8e %.8e %.8e %.8e"
dump_modify 1 first yes
run  ${{nsteps}}
"""

        lj_style = (
            f"pair_style lj/cut {self.lj_params['cutoff']}"
            if self.lj_active
            else "pair_style none"
        )
        if self.lj_active:
            lj_12_factor = 0.0 if self.lj_params.get("exclude_bonded", False) else 1.0
        else:
            lj_12_factor = 0.0
        init_script_template: str = f"""
# NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
# variable up equal 1.0e-1
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
# variable atomjiggle equal 1.0e-6

# Uncomment one of these blocks, depending on what units
# you are using in LAMMPS and for output

# metal units, elastic constants in eV/A^3
#units		metal
#variable cfac equal 6.2414e-7
#variable cunits string eV/A^3

# Choose potential
# metal units, elastic constants in GPa
units		metal
atom_style	full
{lj_style}
#pair_style      lj/cut 1.0 
#pair_style      yukawa 1.0 2.0  
#pair_modify     mix arithmetic
bond_style      harmonic
angle_style     harmonic
dihedral_style  zero
special_bonds   amber

# First weight coefficient (0.0 or 1.0) controls whether LJ interactions are used for bonded beads
# special_bonds lj 0.0 1.0 1.0 excludes LJ for 1-2 neighbors (bonded pairs)
# see https://docs.lammps.org/special_bonds.html
special_bonds   lj {lj_12_factor} 1.0 1.0

variable cfac equal 1.0e-4
variable cunits string GPa
boundary	p p p

# Need to set mass to something, just to satisfy LAMMPS
#mass 1 1.0e-20
read_data {self.network_filename}

# real units, elastic constants in GPa
#units		real
#variable cfac equal 1.01325e-4
#variable cunits string GPa

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 5.0e-10
variable maxiter equal 100000
variable maxeval equal 20000
variable dmax equal 1.0e-2
"""

        lj_coeff = (
            f"pair_coeff * * {self.lj_params['epsilon']} {self.lj_params['sigma']}"
            if self.lj_active
            else ""
        )
        shift = "pair_modify shift yes" if self.lj_active else ""
        potential_script_template: str = f"""
# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Setup neighbor style
neighbor 3.0 nsq
neigh_modify once no every 1 delay 0 check yes
comm_modify cutoff 2.0

{lj_coeff}
{shift}
# Setup minimization style
min_style	     cg
min_modify	     dmax ${{dmax}} line quadratic

timestep	0.001


# Setup output
thermo		500
thermo_style custom step temp pe ebond eangle edihed press pxx pyy pzz pxy pxz pyz lx ly lz vol
# thermo_style custom step temp atoms vx vy vz
thermo_modify norm no
"""

        files = {
            "in.deformation": deformation_script_template,
            "init.mod": init_script_template,
            "potential.mod": potential_script_template,
        }

        for filename, content in files.items():
            if directory is None:
                with open(filename, "w", encoding="utf8") as f:
                    f.write(content)
            else:
                path = os.path.join(directory, filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf8") as f:
                    f.write(content)


class ElasticScript:
    network_filename: str

    def __init__(self, network_filename: str):
        self.network_filename = network_filename

    def write_to_file(self, directory: str | None = None):

        main_elastic_script_template = """
include init.mod
include potential.mod

# Compute initial state
fix 3 all box/relax  aniso 0.0
minimize ${etol} ${ftol} ${maxiter} ${maxeval}

variable tmp equal pxx
variable pxx0 equal ${tmp}
variable tmp equal pyy
variable pyy0 equal ${tmp}
# variable tmp equal pzz
# variable pzz0 equal ${tmp}
# variable tmp equal pyz
# variable pyz0 equal ${tmp}
# variable tmp equal pxz
# variable pxz0 equal ${tmp}
variable tmp equal pxy
variable pxy0 equal ${tmp}

variable tmp equal lx
variable lx0 equal ${tmp}
variable tmp equal ly
variable ly0 equal ${tmp}
# variable tmp equal lz
# variable lz0 equal ${tmp}

# These formulas define the derivatives w.r.t. strain components
# Constants uses $, variables use v_ 
# These are the definition of shear components along certain axes,
# i.e., G=dP/(dX/l)
variable d1 equal -(v_pxx1-${pxx0})/(v_delta/v_len0)*${cfac}
variable d2 equal -(v_pyy1-${pyy0})/(v_delta/v_len0)*${cfac}
# variable d3 equal -(v_pzz1-${pzz0})/(v_delta/v_len0)*${cfac}
# variable d4 equal -(v_pyz1-${pyz0})/(v_delta/v_len0)*${cfac}
# variable d5 equal -(v_pxz1-${pxz0})/(v_delta/v_len0)*${cfac}
variable d6 equal -(v_pxy1-${pxy0})/(v_delta/v_len0)*${cfac}

displace_atoms all random ${atomjiggle} ${atomjiggle} ${atomjiggle} 87287 units box

# Write restart
unfix 3
write_restart restart.equil

# uxx Perturbation

variable dir equal 1
include displace.mod

# uyy Perturbation

variable dir equal 2
include displace.mod

# uzz Perturbation

# variable dir equal 3
# include displace.mod

# uyz Perturbation

# variable dir equal 4
# include displace.mod

# uxz Perturbation

# variable dir equal 5
# include displace.mod

# uxy Perturbation

variable dir equal 6
include displace.mod

# Output final values

variable C11all equal ${C11}
variable C22all equal ${C22}
# variable C33all equal ${C33}

variable C12all equal 0.5*(${C12}+${C21})
# variable C13all equal 0.5*(${C13}+${C31})
# variable C23all equal 0.5*(${C23}+${C32})

# variable C44all equal ${C44}
# variable C55all equal ${C55}
variable C66all equal ${C66}

# variable C14all equal 0.5*(${C14}+${C41})
# variable C15all equal 0.5*(${C15}+${C51})
variable C16all equal 0.5*(${C16}+${C61})

# variable C24all equal 0.5*(${C24}+${C42})
# variable C25all equal 0.5*(${C25}+${C52})
variable C26all equal 0.5*(${C26}+${C62})

# variable C34all equal 0.5*(${C34}+${C43})
# variable C35all equal 0.5*(${C35}+${C53})
# variable C36all equal 0.5*(${C36}+${C63})

# variable C45all equal 0.5*(${C45}+${C54})
# variable C46all equal 0.5*(${C46}+${C64})
# variable C56all equal 0.5*(${C56}+${C65})

# Average moduli 
variable C11ave equal (${C11}+${C22})/2.0
# variable C12ave equal ${C12all}
# variable C44cubic equal ${C66all}

variable bulkmodulus equal (${C11ave}+${C12all})/2.0
variable shearmodulus1 equal ${C66} # simple shear
variable shearmodulus2 equal (${C11ave}-${C12all})/2.0 # pure shear
variable poissonratio equal (1.0-(${shearmodulus2}/${bulkmodulus}))/(1.0+(${shearmodulus2}/${bulkmodulus}))

# dump 8 all atom 1000 deform_dump.lammpstrj

# variable C11cubic equal (${C11all}+${C22all}+${C33all})/3.0
# variable C12cubic equal (${C12all}+${C13all}+${C23all})/3.0
# variable C44cubic equal (${C44all}+${C55all}+${C66all})/3.0

# variable bulkmodulus equal (${C11cubic}+2*${C12cubic})/3.0
# variable shearmodulus1 equal ${C44cubic}
# variable shearmodulus2 equal (${C11cubic}-${C12cubic})/2.0
# variable poissonratio equal 1.0/(1.0+${C11cubic}/${C12cubic})
  
# For Stillinger-Weber silicon, the analytical results
# are known to be (E. R. Cowley, 1988):
#               C11 = 151.4 GPa
#               C12 = 76.4 GPa
#               C44 = 56.4 GPa

print "========================================="
print "Components of the Elastic Constant Tensor"
print "========================================="

print "Elastic Constant C11all = ${C11all} ${cunits}"
print "Elastic Constant C22all = ${C22all} ${cunits}"
# print "Elastic Constant C33all = ${C33all} ${cunits}"

print "Elastic Constant C12all = ${C12all} ${cunits}"
# print "Elastic Constant C13all = ${C13all} ${cunits}"
# print "Elastic Constant C23all = ${C23all} ${cunits}"

# print "Elastic Constant C44all = ${C44all} ${cunits}"
# print "Elastic Constant C55all = ${C55all} ${cunits}"
print "Elastic Constant C66all = ${C66all} ${cunits}"

# print "Elastic Constant C14all = ${C14all} ${cunits}"
# print "Elastic Constant C15all = ${C15all} ${cunits}"
print "Elastic Constant C16all = ${C16all} ${cunits}"

# print "Elastic Constant C24all = ${C24all} ${cunits}"
# print "Elastic Constant C25all = ${C25all} ${cunits}"
print "Elastic Constant C26all = ${C26all} ${cunits}"

# print "Elastic Constant C34all = ${C34all} ${cunits}"
# print "Elastic Constant C35all = ${C35all} ${cunits}"
# print "Elastic Constant C36all = ${C36all} ${cunits}"

# print "Elastic Constant C45all = ${C45all} ${cunits}"
# print "Elastic Constant C46all = ${C46all} ${cunits}"
# print "Elastic Constant C56all = ${C56all} ${cunits}"

#print "========================================="
#print "Average properties for a cubic crystal"
#print "========================================="

print "Bulk Modulus = ${bulkmodulus} ${cunits}"
print "Shear Modulus 1 = ${shearmodulus1} ${cunits}"
print "Shear Modulus 2 = ${shearmodulus2} ${cunits}"
print "Poisson Ratio = ${poissonratio}"
"""

        displace_script_template = """
# NOTE: This script should not need to be
# modified. See in.elastic for more info.
#
# Find which reference length to use

variable dt equal 0.01
variable strain equal 1e-4
variable srate equal 1e-5
variable M equal 1000 # averaging time for properties
timestep ${dt}
variable strainsteps equal ${strain}/${dt}/${srate}
variable nsteps equal round(${strainsteps})
print "nsteps ${nsteps}"

if "${dir} == 1" then &
   "variable len0 equal ${lx0}" 
if "${dir} == 2" then &
   "variable len0 equal ${ly0}" 
# if "${dir} == 3" then &
#    "variable len0 equal ${lz0}" 
# if "${dir} == 4" then &
#    "variable len0 equal ${lz0}" 
# if "${dir} == 5" then &
#    "variable len0 equal ${lz0}" 
if "${dir} == 6" then &
   "variable len0 equal ${ly0}" 

# Reset box and simulation parameters

clear
box tilt large
read_restart restart.equil
include potential.mod

# Negative deformation
# variable delta equal -${up}*${len0}
# variable deltaxy equal -${up}*xy
# # variable deltaxz equal -${up}*xz
# # variable deltayz equal -${up}*yz
# if "${dir} == 1" then &
#    "fix 42 all deform 1 x erate -${srate} units box remap x" &
#    "run  ${nsteps}"
# if "${dir} == 2" then &
#    "fix 42 all deform 1 y erate -${srate} units box remap x" &
#    "run  ${nsteps}"
# # if "${dir} == 1" then &
# #    "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"
# # if "${dir} == 2" then &
# #    "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"
# # if "${dir} == 3" then &
# #    "change_box all z delta 0 ${delta} remap units box"
# # if "${dir} == 4" then &
# #    "change_box all yz delta ${delta} remap units box"
# # if "${dir} == 5" then &
# #    "change_box all xz delta ${delta} remap units box"
# if "${dir} == 6" then &
#    "fix 42 all deform 1 xy erate -${srate} units box remap x" &
#    "run  ${nsteps}"

# # Relax atoms positions

# minimize ${etol} ${ftol} ${maxiter} ${maxeval}


# fix 69 all ave/time 1 $M $M v_pxx v_pyy v_pzz v_pxy v_pxz v_pyz
# Obtain new stress tensor
# variable tmp equal pxx
# variable pxx1 equal ${tmp}
# variable tmp equal pyy
# variable pyy1 equal ${tmp}
# # variable tmp equal pzz
# # variable pzz1 equal ${tmp}
# variable tmp equal pxy
# variable pxy1 equal ${tmp}
# # variable tmp equal pxz
# # variable pxz1 equal ${tmp}
# # variable tmp equal pyz
# # variable pyz1 equal ${tmp}

# # Compute elastic constant from pressure tensor

# variable C1neg equal ${d1}
# variable C2neg equal ${d2}
# # variable C3neg equal ${d3}
# # variable C4neg equal ${d4}
# # variable C5neg equal ${d5}
# variable C6neg equal ${d6}

# Reset box and simulation parameters

clear
box tilt large
read_restart restart.equil
include potential.mod

# Positive deformation

variable delta equal ${up}*${len0}
variable deltaxy equal ${up}*xy
# variable deltaxz equal ${up}*xz
# variable deltayz equal ${up}*yz
if "${dir} == 1" then &
   "fix 42 all deform 1 x erate ${srate} units box remap x" &
   "run  ${nsteps}"
if "${dir} == 2" then &
   "fix 42 all deform 1 y erate ${srate} units box remap x" &
   "run  ${nsteps}"
# if "${dir} == 1" then &
#    "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"
# if "${dir} == 2" then &
#    "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"
# if "${dir} == 3" then &
#    "change_box all z delta 0 ${delta} remap units box"
# if "${dir} == 4" then &
#    "change_box all yz delta ${delta} remap units box"
# if "${dir} == 5" then &
#    "change_box all xz delta ${delta} remap units box"
if "${dir} == 6" then &
   "fix 42 all deform 1 xy erate ${srate} units box remap x" &
   "run  ${nsteps}"

# Relax atoms positions
minimize ${etol} ${ftol} ${maxiter} ${maxeval}

# Obtain new stress tensor
variable tmp equal pe
variable e1 equal ${tmp}
variable tmp equal press
variable p1 equal ${tmp}
variable tmp equal pxx
variable pxx1 equal ${tmp}
variable tmp equal pyy
variable pyy1 equal ${tmp}
# variable tmp equal pzz
# variable pzz1 equal ${tmp}
variable tmp equal pxy
variable pxy1 equal ${tmp}
# variable tmp equal pxz
# variable pxz1 equal ${tmp}
# variable tmp equal pyz
# variable pyz1 equal ${tmp}

# Compute elastic constant from pressure tensor
variable C1pos equal ${d1}
variable C2pos equal ${d2}
# variable C3pos equal ${d3}
# variable C4pos equal ${d4}
# variable C5pos equal ${d5}
variable C6pos equal ${d6}

# Combine positive and negative 
variable C1${dir} equal ${C1pos}
variable C2${dir} equal ${C2pos}
# variable C3${dir} equal 0.5*(${C3neg}+${C3pos})
# variable C4${dir} equal 0.5*(${C4neg}+${C4pos})
# variable C5${dir} equal 0.5*(${C5neg}+${C5pos})
variable C6${dir} equal ${C6pos}

# variable C1${dir} equal 0.5*(${C1neg}+${C1pos})
# variable C2${dir} equal 0.5*(${C2neg}+${C2pos})
# variable C3${dir} equal 0.5*(${C3neg}+${C3pos})
# variable C4${dir} equal 0.5*(${C4neg}+${C4pos})
# variable C5${dir} equal 0.5*(${C5neg}+${C5pos})
# variable C6${dir} equal 0.5*(${C6neg}+${C6pos})

# Delete dir to make sure it is not reused
variable dir delete

"""

        init_script_template = f"""
# NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 1e-4
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
variable atomjiggle equal 1.0e-5

# Uncomment one of these blocks, depending on what units
# you are using in LAMMPS and for output

# metal units, elastic constants in eV/A^3
#units		metal
#variable cfac equal 6.2414e-7
#variable cunits string eV/A^3

# metal units, elastic constants in GPa
# units		metal
# variable cfac equal 1.0e-4
# variable cunits string GPa

# real units, elastic constants in GPa
#units		real
#variable cfac equal 1.01325e-4
#variable cunits string GPa

# COPIED FROM MODIFIED SCRIPT
#---------------------------------
# Choose potential
# metal units, elastic constants in GPa
units		metal
atom_style	full
bond_style      harmonic
angle_style     harmonic
dihedral_style  zero
special_bonds   amber
variable cfac equal 1.0e-4
variable cunits string GPa
boundary	p p p
#---------------------------------
read_data {self.network_filename}

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 1.0e-10
variable maxiter equal 100
variable maxeval equal 1000
variable dmax equal 1.0e-2

# Need to set mass to something, just to satisfy LAMMPS
mass 1 1.0e-20


"""

        potential_script_template = """
# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# EXAMPLE SCRIPT
#------------------------------------------------------
# # Choose potential
# pair_style	sw
# pair_coeff * * Si.sw Si

# # Setup neighbor style
# neighbor 1.0 nsq
# neigh_modify once no every 1 delay 0 check yes

# # Setup minimization style
# min_style	     cg
# min_modify	     dmax ${dmax} line quadratic

# # Setup output
# thermo		1
# thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
# thermo_modify norm no

# COPIED FROM MODIFIED SCRIPT
#------------------------------------------------------
# Setup neighbor style
neighbor 3.0 nsq
neigh_modify once no every 1 delay 0 check yes
comm_modify cutoff 6.0

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

timestep	0.001


# Setup output
thermo		1000
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no

"""

        files = {
            "in.elastic": main_elastic_script_template,
            "displace.mod": displace_script_template,
            "init.mod": init_script_template,
            "potential.mod": potential_script_template,
        }

        for filename, content in files.items():
            if directory is None:
                with open(filename, "w", encoding="utf8") as f:
                    f.write(content)
            else:
                path = os.path.join(directory, filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf8") as f:
                    f.write(content)
