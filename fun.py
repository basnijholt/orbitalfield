# Functions library used to calculate phase diagrams
# By Bas Nijholt and Anton Akhmerov

# 1. Standard library imports

import os.path
from math import pi
from itertools import product
# 2. External package imports
import kwant
import h5py
import types
import sympy
from sympy.physics.quantum import TensorProduct as kr
import numpy as np
import holoviews as hv
import scipy.sparse.linalg as sla
from scipy.constants import hbar, m_e, eV, physical_constants
# 3. Internal imports
from Bloch import BlochSpherePlot, BlochSphere
from discretizer import Discretizer, momentum_operators

__all__ = ['BlochSphere', 'BlochSpherePlot', 'bnds', 'constants',
           'create_holoviews', 'create_mask', 'dimensions', 'find_decay_length',
           'find_gap', 'load_data', 'make_3d_wire', 'make_3d_wire_external_sc',
           'make_params', 'modes', 'find_phase_bounds', 'nearest', 'save_data',
           'slowest_evan_mode', 'spherical_coords']

sx, sy, sz = [sympy.physics.matrices.msigma(i) for i in range(1, 4)]
s0 = sympy.eye(2)
s0sz = np.kron(s0, sz)
s0s0 = np.kron(s0, s0)

# Parameters taken from arXiv:1204.2792
# All constant parameters, mostly fundamental constants, in a SimpleNamespace.
constants = types.SimpleNamespace(
    m=0.015 * m_e,  # effective mass in kg
    a=10,  # lattice spacing in nm
    g=50,  # Lande factor
    hbar=hbar,
    m_e=m_e,
    e=eV,
    eV=eV,
    meV=eV * 1e-3)

constants.t = (hbar ** 2 / (2 * constants.m)) * (1e18 / constants.meV)  # meV * nm^2
constants.mu_B = physical_constants['Bohr magneton'][0] / constants.meV

# All frequently used dimensions in a SimpleNamespace.
dimensions = types.SimpleNamespace(
    B=hv.Dimension(name=('B', r'$B$'), unit='T'),
    mu=hv.Dimension(name=('mu', r'$\mu$'), unit='meV'),
    gap=hv.Dimension(name='Band gap', unit=r'$\mu$eV'),
    # gap=hv.Dimension(name=r'$E_\textrm{gap}$', unit=r'\textmu eV'),
    decay_length=hv.Dimension(name='Inverse decay length', unit=r'$\mu m^{-1}$'),
    # decay_length=hv.Dimension(r'$\xi^{-1}$', unit=r'\textmu m$^{-1}$'),
    k=hv.Dimension(name=r'$k$', unit=r'nm$^{-1}$'),
    E=hv.Dimension(name=r'$E$', unit='meV'),
    delta=hv.Dimension(name=('Delta', r'$\Delta$'), unit='mev'),
    delta_ind=hv.Dimension(name=('Delta_ind', r'$\Delta_{ind}$'), unit='mev'),
    angles={'kdims': [hv.Dimension(name=('theta', r'$\theta$'), unit=r'$\pi$'),
                      hv.Dimension(name=('phi', r'$\phi$'), unit=r'$\pi$')]})


def spherical_coords(r, theta, phi):
    """Transform spherical coordinates to Cartesian."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    # last term is to ensure x, y and z have same dimensions
    z = r * np.cos(theta) + 0 * phi
    return np.array([x, y, z]).T


def make_params(alpha=20,
                B_x=0,
                B_y=0,
                B_z=0,
                Delta=0.25,
                mu=0,
                orbital=True,
                A_correction=True,
                t=constants.t,
                g=constants.g,
                mu_B=constants.mu_B,
                V=lambda x,y,z: 0,
                **kwargs):
    """Function that creates a namespace with parameters.

    Parameters:
    -----------
    alpha : float
        Spin-orbit coupling strength in units of meV*nm.
    B_x, B_y, B_z : float
        The magnetic field strength in the x, y and z direction in units of Tesla.
    Delta : float
        The superconducting gap in units of meV.
    mu : float
        The chemical potential in units of meV.
    orbital : bool
        Switches the orbital effects on and off.
    A_correction : bool
        Corrects for the net supercurrent flowing in the wire. If True, the
        current will be set to zero.
    t : float
        Hopping parameter in meV * nm^2.
    g : float
        Lande g factor.
    mu_B : float
        Bohr magneton in meV/K.
    V : function
        Function of spatial coordinates (x, y, z) with is added to mu.

    Returns:
    --------
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    """
    p = types.SimpleNamespace(t=t,
                              g=g,
                              mu_B=mu_B,
                              alpha=alpha,
                              B_x=B_x,
                              B_y=B_y,
                              B_z=B_z,
                              Delta=Delta,
                              mu=mu,
                              orbital=orbital,
                              A_correction=A_correction,
                              V=V,
                              **kwargs)
    return p

# Creating the system

def make_3d_wire(a=10, R=50, L=None, holes=True, verbose=False):
    """Makes a hexagonal shaped 3D wire.

    Parameters:
    -----------
    a : int
        Lattice constant in nm.
    R : int
        Radius of the wire in units in units of nm.
    L : int
        Length of the wire in units of nm, L=None if infinite wire.
    holes : bool
        True if PHS, False if no holes and only in spin space.
    vebose : bool
        Prints the discretized Hamiltonian.

    Returns:
    --------
    sys : kwant.builder.(In)finiteSystem object
        The finalized (in)finite system.
    """
    k_x, k_y, k_z = momentum_operators
    t, B_x, B_y, B_z, mu_B, Delta, mu, alpha, g, V = sympy.symbols('t B_x B_y B_z mu_B Delta mu alpha g V', real=True)
    k =  sympy.sqrt(k_x**2+k_y**2+k_z**2)
    if holes:
        hamiltonian = ((t * k**2 - mu - V) * kr(s0, sz) +
                       alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
                       0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
                       Delta * kr(s0, sx))
    else:
        hamiltonian = ((t * k**2 - mu - V) * s0 + alpha * (k_y * sx - k_x * sy) +
                       0.5 * g * mu_B * (B_x * sx + B_y * sy + B_z * sz) +
                       Delta * s0)

    tb = Discretizer(hamiltonian, space_dependent={'V'}, lattice_constant=a,
                     verbose=verbose)
    sys = kwant.Builder(kwant.TranslationalSymmetry((-a, 0, 0)))

    if L is None:
        L = 1

    def hexagon(pos):
        (x, y, z) = pos
        return (y > -R and y < R and y > -2 * (R - z) and y < -2 *
                (z - R) and y < 2 * (z + R) and y > -2 *
                (z + R) and x >= 0 and x < L)

    sys[tb.lattice.shape(hexagon, (0, 0, 0))] = tb.onsite

    def peierls(val, ind):
        def phase(s1, s2, p):
            x, y, z = s1.pos
            A = lambda p, x, y, z: [p.B_y * z - p.B_z * y, 0, p.B_x * y]
            A_site = A(p, x, y, z)[ind]
            A_site *= a * 1e-18 * eV / hbar
            if holes:
                return np.cos(A_site) * s0s0 - 1j * np.sin(A_site) * s0sz
            else:
                return np.exp(-1j * A_site)

        def with_phase(s1, s2, p):
            if p.orbital:
                try:
                    return phase(s1, s2, p).dot(val(s1, s2, p))
                except AttributeError:
                    return phase(s1, s2, p) * val(s1, s2, p)
            else:
                return val(s1, s2, p)
        return with_phase


    for hop, val in tb.hoppings.items():
        ind = np.argmax(hop.delta)
        sys[hop] = peierls(val, ind)
    return sys.finalized()



def make_3d_wire_external_sc(a=10, r1=50, r2=70, phi=135, angle=45, finalized=True):
    """Makes a hexagonal shaped 3D wire with external superconductor.

    Parameters:
    -----------
    a : int
        Lattice constant in nm.
    r1 : float
        Diameter of wire part in nm.
    r2 : float
        Diameter of wire plus superconductor part in nm.
    phi : float
        Coverage angle of superconductor in degrees.
    angle : float
        Angle of the superconductor w.r.t. the y-axis in degrees.
    finalized : bool
        Return a finalized system if True or kwant.Builder object
        if False.

    Returns:
    --------
    sys : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    """
    k_x, k_y, k_z = momentum_operators
    t, B_x, B_y, B_z, mu_B, Delta, mu, alpha, g, V = sympy.symbols('t B_x B_y B_z mu_B Delta mu alpha g V', real=True)
    t_interface = sympy.symbols('t_interface', real=True)
    k =  sympy.sqrt(k_x**2+k_y**2+k_z**2)

    hamiltonian = ((t * k**2 - mu - V) * kr(s0, sz) +
                   alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
                   0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
                   Delta * kr(s0, sx))

    def cylinder_sector(r1, r2=0, phi=360, angle=angle):
        phi *= np.pi / 360
        angle *= np.pi / 180
        r1sq, r2sq = r1 ** 2, r2 ** 2
        def sector(pos):
            x, y, z = pos
            n = (y + 1j * z) * np.exp(1j * angle)
            y, z = n.real, n.imag
            rsq = y ** 2 + z ** 2
            return r2sq <= rsq < r1sq and z >= np.cos(phi) * np.sqrt(rsq)
        r_mid = (r1 + r2) / 2
        return sector, (0, r_mid * np.sin(angle), r_mid * np.cos(angle))

    args = dict(space_dependent={'V'}, lattice_constant=a)
    tb_normal = Discretizer(hamiltonian.subs(Delta, 0), **args)
    tb_sc = Discretizer(hamiltonian, **args)
    tb_interface = Discretizer(hamiltonian.subs(t, t_interface), **args)
    lat = tb_normal.lattice
    sys = kwant.Builder(kwant.TranslationalSymmetry((-a, 0, 0)))

    shape_normal = cylinder_sector(r1=r1, angle=angle)
    shape_sc = cylinder_sector(r1=r2, r2=r1, phi=phi, angle=angle)

    sys[lat.shape(*shape_normal)] = tb_normal.onsite
    sys[lat.shape(*shape_sc)] = tb_sc.onsite
    sc_sites = list(sys.expand(lat.shape(*shape_sc)))

    def peierls(val, ind):
        def phase(s1, s2, p):
            x, y, z = s1.pos
            A = lambda p, x, y, z: [p.B_y * z - p.B_z * y, 0, p.B_x * y]
            A_site = A(p, x, y, z)[ind]
            if p.A_correction:
                A_sc = [A(p, *site.pos) for site in sc_sites]
                A_site -= np.mean(A_sc, axis=0)[ind]
            A_site *= a * 1e-18 * eV / hbar
            return np.cos(A_site) * s0s0 - 1j * np.sin(A_site) * s0sz

        def with_phase(s1, s2, p):
            if p.orbital:
                return phase(s1, s2, p).dot(val(s1, s2, p))
            else:
                return val(s1, s2, p)
        return with_phase

    for hop, val in tb_sc.hoppings.items():
        ind = np.argmax(hop.delta)
        sys[hop] = peierls(val, ind)

    def at_interface(site1, site2):
        return ((shape_sc[0](site1.pos) and shape_normal[0](site2.pos)) or
                (shape_normal[0](site1.pos) and shape_sc[0](site2.pos)))

    # Hoppings at the barrier between wire and superconductor
    for hop, val in tb_interface.hoppings.items():
        hopping_iterator = ((i, j) for (i, j) in kwant.builder.HoppingKind(hop.delta, lat)(sys) if at_interface(i, j))
        ind = np.argmax(hop.delta)
        sys[hopping_iterator] = peierls(val, ind)
    if finalized:
        return sys.finalized()
    else:
        return sys


# Phase diagram

def find_phase_bounds(lead, p, B, k=0, num_bands=20):
    """Find the phase boundaries.

    Solve an eigenproblem that finds values of chemical potential at which the
    gap closes at momentum k=0. We are looking for all real solutions of the
    form H\psi=0 so we solve sigma_0 * tau_z H * psi = mu * psi.

    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    B : tuple of floats
        A tuple that contains magnetic field strength in x, y and z-directions.
    k : float
        Momentum value, by default set to 0.

    Returns:
    --------
    chemical_potential : numpy array
        Twenty values of chemical potential at which a bandgap closes at k=0.
    """
    p.B_x, p.B_y, p.B_z = B
    h, t = lead.cell_hamiltonian(args=[p]), lead.inter_cell_hopping(args=[p])
    tk = lambda k: t * np.exp(1j * k)
    h_k = lambda k: h + tk(k) + tk(k).T.conj()
    sigma_z = np.array([[1, 0], [0, -1]])
    chemical_potentials = np.kron(np.eye(len(h) // 2), sigma_z) @ h_k(k)
    return sla.eigs(chemical_potentials, k=num_bands, sigma=0)[0]


def find_gap(lead, p, val, tol=1e-3):
    """Finds the gapsize by peforming a binary search of the modes with a
    tolarance of tol.

    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    val : tuple
        An array that contains the value of (B, mu, point), the values at
        which this function is evaluated.
    tol : float
        The precision of the binary search.

    Returns:
    --------
    gap : float
        Size of the gap.
    """
    def gap_minimizer(lead, p, energy):
        """Function that minimizes a function to find the band gap.

        This objective function checks if there are progagating modes at a
        certain energy. Returns zero if there is a propagating mode.

        Parameters:
        -----------
        energy : float
            Energy at which this function checks for propagating modes.
        h0 : numpy array
            Onsite Hamiltonian, sys.cell_hamiltonian(args=[p])
        t0 : numpy array
            Hopping matrix, sys.inter_cell_hopping(args=[p])

        Returns:
        --------
        minimized_scalar : float
            Value that is zero when there is a propagating mode.
        """
        h, t = lead.cell_hamiltonian(args=[p]), lead.inter_cell_hopping(args=[p])
        ev = modes(h - energy * np.identity(len(h)), t)
        norm = (ev * ev.conj()).real
        return np.sort(np.abs(norm - 1))[0]

    B, mu, point = val

    # These points are not within the topological boundaries
    if not point:
        return np.nan
    else:
        p.B_x, p.B_y, p.B_z = B
        p.mu = mu
        bands = kwant.physics.Bands(lead, args=[p])
        band_k_0 = np.abs(bands(k=0)).min()
        lim = [0, band_k_0]
        if gap_minimizer(lead, p, energy=0) < 1e-15:
            gap = 0
        else:
            while lim[1] - lim[0] > tol:
                energy = sum(lim) / 2.
                par = gap_minimizer(lead, p, energy)
                if par < 1e-10:
                    lim[1] = energy
                else:
                    lim[0] = energy
            gap = sum(lim) / 2.
        return gap


def slowest_evan_mode(lead, p, c=constants):
    """Find the slowest decaying (evanescent) mode.

    It uses an adapted version of the function kwant.physics.leads.modes,
    in such a way that it returns the eigenvalues of the translation operator
    (lamdba = e^ik). The imaginary part of the wavevector k, is the part that
    makes it decay. The inverse of this Im(k) is the size of a Majorana bound
    state. The norm of the eigenvalue that is closest to one is the slowes
    decaying mode. Also called decay length.

    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    c : types.SimpleNamespace object
        A namespace container with all constant (fundamental) parameters used.

    Returns:
    --------
    majorana_length : float
        The length of the Majorana.
    """
    h, t = lead.cell_hamiltonian(args=[p]), lead.inter_cell_hopping(args=[p])
    ev = modes(h, t)
    norm = ev * ev.conj()
    idx = np.abs(norm - 1).argmin()
    majorana_length = np.abs(c.a / np.log(ev[idx]).real)
    return majorana_length


def find_decay_length(lead, p, val, c=constants):
    """Finds the slowest decaying (evanescent) mode.

    Calls slowest_evan_mode()
    and evaluates it on val.

    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    val : tuple
        An array that contains the value of (B, mu, point), the values at
        which this function is evaluated.
    c : types.SimpleNamespace object
        A namespace container with all constant (fundamental) parameters used.

    Returns:
    --------
    decay_length : float
        The length of the Majorana or decay lentgh.
    """
    B, mu, point = val
    # These points are not within the topological boundaries
    if not point:
        return np.nan
    else:
        p.B_x, p.B_y, p.B_z = B
        p.mu = mu
        return slowest_evan_mode(lead, p)


def create_mask(Bs, thetas, phis, mu_mesh, mus_output):
    """Creates a mask inside the topological boundaries from find_phase_bounds.

    See Stackexchange question:
    http://stackoverflow.com/questions/29694087/from-1d-graph-to-2d-mask

    Parameters:
    -----------
    Bs : numpy array
        The magnetic field strength in units of Tesla.
    thetas : numpy array
        The polar angles, for the direction of magnetic field.
    phis : numpy array
        The azimuthal  angles, for the direction of magnetic field.
    mu_mesh : numpy array
        A linear space for values of chemical potential.
    mus_output : numpy array
        Values of chemical potential of where the boundary of the topological
        regions lie.
        Output of find_phase_bounds()

    Returns:
    --------
    mus : numpy array
        Values of chemical potential for different values of magnetic field
        and angles.
        In the same shape as mask.
    vals : numpy array
        A long array with all values of (B, mu, point) with shape (-1, 3).
    mask : numpy array
        mask that corresponds to the topological region. In between the odd and
        even lines. True or False for different values of magnetic field and
        angles. Same shape as mus.
    """
    pos = spherical_coords(
        Bs.reshape(-1, 1, 1), thetas.reshape(1, -1, 1), phis.reshape(1, 1, -1))

    pos_vec = pos.reshape(-1, 3)

    mus = np.reshape(mus_output, (len(phis), len(thetas), len(Bs), -1))
    mus = np.sort(mus, axis=-1)

    mus_non_degenerate = np.array(mus[:, :, :, ::2]).real  # np.array makes copy
    topo_region_limits = mus_non_degenerate.reshape(
        len(phis), len(thetas), len(Bs), -1, 2)

    mus[abs(mus.imag) > 1e-10] = None  # get rid of propagating modes.

    mus = np.array(mus.real)  # np.array makes copy

    X, Y = np.meshgrid(Bs, mu_mesh)
    low = topo_region_limits[:, :, :, :, 0]
    up = topo_region_limits[:, :, :, :, 1]

    # makes a mask that covers the topological areas see my Stackexchange
    # question
    mask = np.any((low[:, :, :, None, :] <= Y.T[None, None, :, :, None]) &
                  (up[:, :, :, None, :] >= Y.T[None, None, :, :, None]),
                  axis=-1)

    if thetas[0] == 0:
        mask[1:, 0] = False  # because if theta = 0 it's independent of phi

    mask_vec = mask.reshape(-1, len(mu_mesh))
    vals = [(B, mu, point) for points, B in zip(mask_vec, pos_vec) for point, mu
            in zip(points, mu_mesh)]
    return mus, vals, mask


# Saving and loading data

def save_data(fname, Bs, thetas, phis, mu_mesh, mus_output, gaps_output,
              decay_length_output, p,
              c=constants):
    """Saves data to h5f format.

    Parameters:
    -----------
    fname : str
        Filename of the HDF5 file.
    Bs : numpy array
        The magnetic field strength in units of Tesla.
    thetas : numpy array
        The polar angles, for the direction of magnetic field.
    phis : numpy array
        The azimuthal  angles, for the direction of magnetic field.
    mu_mesh : numpy array
        A linear space for values of chemical potential.
    mus_output : numpy array
        Values of chemical potential of where the boundary of the topological
        regions lie.
        Output of find_phase_bounds()
    gaps_output : numpy array
        Values of band gap of all angles, values of magnetic field, and mus.
    decay_length_output : numpy array
        Values of Majorana length of all angles, values of magnetic field, and
        mus.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    c : types.SimpleNamespace object
        A namespace container with all constant (fundamental) parameters used.
    """
    if os.path.isfile(fname):
        print('File already exists.')
    else:
        h5f = h5py.File(fname, 'w')
        h5f.create_dataset('Bs', data=Bs)
        h5f.create_dataset('thetas', data=thetas)
        h5f.create_dataset('phis', data=phis)
        h5f.create_dataset('mu_mesh', data=mu_mesh)
        h5f.create_dataset('mus_output',
                           data=mus_output,
                           compression="gzip",
                           compression_opts=9)
        h5f.create_dataset('gaps_output',
                           data=gaps_output,
                           compression="gzip",
                           compression_opts=9)
        h5f.create_dataset('decay_length_output',
                           data=decay_length_output,
                           compression="gzip",
                           compression_opts=9)

        parameters = {k: v for k, v in list(vars(c).items()) + list(vars(p).items())}
        for k, v in list(parameters.items()):
            if isinstance(v, int) or isinstance(v, float):
                h5f.create_dataset('parameters/' + k, data=v)
        h5f.close()


def load_data(fname):
    """Loads data from h5py file.

    Loads data that is saved with the save_data() function.

    Parameters:
    -----------
    fname : str
        Filename of the HDF5 file.

    Returns:
    --------
    Bs : numpy array
        The magnetic field strength in units of Tesla.
    thetas : numpy array
        The polar angles, for the direction of magnetic field.
    phis : numpy array
        The azimuthal  angles, for the direction of magnetic field.
    mu_mesh : numpy array
        A linear space for values of chemical potential.
    mus : numpy array
        Values of chemical potential of where the boundary of the topological
        regions lie.
        Output of find_phase_bounds()
    gaps : numpy array
        Values of band gap of all angles, values of magnetic field, and mus.
    decay_lengths : numpy array
        Values of Majorana length of all angles, values of magnetic field, and
        mus.
    cdims : dict
        Dictionary that stores all constant parameters used in simulation.
    """
    h5f = h5py.File(fname, 'r')
    Bs = h5f['Bs'][:]
    thetas = h5f['thetas'][:]
    phis = h5f['phis'][:]
    mu_mesh = h5f['mu_mesh'][:]
    mus_output = h5f['mus_output'][:]
    gaps_output = h5f['gaps_output'][:]
    decay_length_output = h5f['decay_length_output'][:]
    cdims = {str(k): v.value for k, v in list(h5f['parameters'].items())}
    h5f.close()

    mus, vals, mask = create_mask(Bs, thetas, phis, mu_mesh, mus_output)

    gaps = np.reshape(gaps_output, mask.shape)
    gaps[1:, 0] = gaps[0, 0]

    decay_lengths = np.reshape(decay_length_output, mask.shape)
    decay_lengths[1:, 0] = decay_lengths[0, 0]

    return (Bs, thetas, phis, mu_mesh, mus, gaps, decay_lengths, cdims)


# HoloViews-related functions.

def create_holoviews(fname, d=dimensions):
    """Creates a HoloViews object.

    Loads data from HDF5 file stored with save_data() and turns it into a
    HoloViews object.

    Parameters:
    -----------
    fname : str
        Filename of the HDF5 file.
    d : types.SimpleNamespace
        A namespace container with all frequently used dimensions.

    Returns:
    --------
    l : holoviews.core.layout.Layout
        Object that contains all data and is formatted with correct metadata and
        dimensions.
    """
    Bs, thetas, phis, mu_mesh, mus, _gaps, _decay_lengths, cdims = load_data(fname)
    bounds = (Bs.min(), mu_mesh.min(), Bs.max(), mu_mesh.max())
    angles = list(product(enumerate(phis), enumerate(thetas)))

    gap_dim = hv.Dimension(('E_gap', r'$E_\mathrm{gap}$'), unit=r'\textmu eV')
    kwargs = {'kdims': [d.B, d.mu],
              'vdims': [gap_dim],
              'bounds': bounds,
              'label': 'Band gap',
              'group': 'Im'}

    gaps = {(theta / pi, phi / pi): hv.Image(1000 * np.rot90(_gaps[i, j]), **kwargs)
            for (i, phi), (j, theta) in angles}

    decay_length_dim = hv.Dimension(('Inverse_decay_length', r'$\xi^{-1}$'), unit=r'\textmu m$^{-1}$')
    kwargs = {'kdims': [d.B, d.mu],
              'vdims': [decay_length_dim],
              'bounds': bounds,
              'label': 'Inverse decay length',
              'group': 'Im'}

    decay_lengths = {(theta / pi, phi / pi): hv.Image(1e3 / np.rot90(_decay_lengths[i, j]), **kwargs)
                     for (i, phi), (j, theta) in angles}  # in 1 / \mu m

    kwargs = {'kdims': [d.B, d.mu],
              'extents': bounds,
              'label': 'Topological boundaries',
              'group': 'Lines'}

    boundaries = {(theta / pi, phi / pi): hv.Path((Bs, mus[i, j, :, ::2]), **kwargs)
                  for (i, phi), (j, theta) in angles}

    BlochSpherePlot.bgcolor = 'white'

    sphere = {(theta / pi, phi / pi): BlochSphere([[1, 0, 0], spherical_coords(1, theta, phi)], group='Sphere')
              for (i, phi), (j, theta) in angles}

    hms = [gaps, decay_lengths, boundaries, sphere]

    l = hv.Layout([hv.HoloMap(hm, **d.angles) for hm in hms])

    l += (l.Im.Band_gap
          * l.Lines.Topological_boundaries).relabel(d.gap.name, group='Phase diagram')

    l += (l.Im.Inverse_decay_length
          * l.Lines.Topological_boundaries).relabel(d.decay_length.name, group='Phase diagram')

    l.cdims = cdims

    # remove non constant dimensions in simulation
    [l.cdims.pop(k) for k in ['B_x', 'B_y', 'B_z', 'mu']]
    return l


def nearest(a, a0=0):
    "Element in nd array `a` closest to the scalar value `a0`"
    if isinstance(a, dict):
        a = np.array(list(a.keys()))
    else:
        a = np.array(a)
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]


def bnds(x, y):
    "Small helper function that return the bounds in (lbrt) format"
    return (min(x), min(y), max(x), max(y))


# Physics functions.

def modes(h_cell, h_hop, tol=1e6):
    """Compute the eigendecomposition of a translation operator of a lead.

    Adapted from kwant.physics.leads.modes such that it returns the eigenvalues.

    Parameters:
    ----------
    h_cell : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    h_hop : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.

    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=exp(i*k).
    """
    m = h_hop.shape[1]
    n = h_cell.shape[0]

    if (h_cell.shape[0] != h_cell.shape[1] or
            h_cell.shape[0] != h_hop.shape[0]):
        raise ValueError("Incompatible matrix sizes for h_cell and h_hop.")

    # Note: np.any(h_hop) returns (at least from numpy 1.6.1 - 1.8-devel)
    #       False if h_hop is purely imaginary
    if not (np.any(h_hop.real) or np.any(h_hop.imag)):
        v = np.empty((0, m))
        return (kwant.physics.PropagatingModes(np.empty((0, n)), np.empty((0,)),
                                               np.empty((0,))),
                kwant.physics.StabilizedModes(np.empty((0, 0)),
                                              np.empty((0, 0)), 0, v))

    # Defer most of the calculation to helper routines.
    matrices, v, extract = kwant.physics.leads.setup_linsys(
        h_cell, h_hop, tol, None)
    ev = kwant.physics.leads.unified_eigenproblem(*(matrices + (tol,)))[0]

    return ev
