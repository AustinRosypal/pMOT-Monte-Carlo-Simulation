\section{Zeeman Effect}
Now that we have Doppler cooling and a magnetic field established, it is time to simulate the Zeeman effect and expose the hyperfine structure of Rb-87.

Remember, for dipole transitions, starting in $F=2$ can transition to $F'=1,2,3$ and starting in $F=1$ can transition to $F'=0,1,2$.  Account for all possible transitions when relevant, using branching ratios or transition probabilities.

\subsection{0 - Precompute Transition Frequency}
We must first precompute the transition frequencies $\omega_{FF'}^{(0)}$ for each allowed transition $|F, m_F\rangle \rightarrow |F', m_F'\rangle$, in the zero-field assumption, and store them so we don't redundantly calculate them.

That makes the $m_F$-dependent resonance at position $\vec{r}$:

\begin{equation}
    \omega_{F, m_F \rightarrow F', m_F'} = \omega_{FF'}^{(0)} + \Delta \omega_Z (\vec{r})
\end{equation}

where 

\begin{equation}
    \Delta \omega_Z = \frac{\mu_B B(\vec{r})}{\hbar}(g_{F'}m_F' - g_Fm_F)
\end{equation}

So, the Lande g-factors must also be known.

The zero-field transition frequencies for the cases of interest are listed in the following table:

\begin{tabular}{|c|c|c|c|}
    \hline
    \textbf{Transition} & \textbf{Name/Usage} & \textbf{Frequency (THz)} & \textbf{Angular Frequency (rad/s)} \\
    \hline
     $F=2 \rightarrow F'=3$ & Cooling Cycle & 384.228115203271 & 2.414176448e15 \\
     $F=2 \rightarrow F'=2$ & Off-resonant Cooling Cycle & 384.227848551271 & 2.414174773e15 \\
     $F=2 \rightarrow F'=1$ & Off-resonant Cooling Cycle & 384.227691610771 & 2.414173787e15 \\
     $F=1 \rightarrow F'=2$ & Repump Cycle & 384.234683233882 & 2.414217716e15 \\
     \hline
\end{tabular}

Regarding the Lande g-factors: For the $5s_{1/2}$ $F=1$ hyperfine manifold, use $g_F = -1/2$.  For the $5s_{1/2}$ $F=2$ hyperfine manifold, use $g_F = +1/2$.  For the $5p_{3/2}$ $F=1,2,3$ hyperfine manifolds, use $g_F = +2/3$.

\subsection{1 - Initial State}
Initialize each simulated atom in a hyperfine state.  Let there be a $3/8$ probability to start in $F=1$ and $5/8$ probability to start in $F=2$.  From there, random probability to begin in any of the Zeeman sublevels within the hyperfine level it is assigned to.  So for example, the initial random selector labels the atom to be in $F=2$, then the random selector labels that atom to be in $m_F = +1$ making the initialized state $|F=2, m_F=+1\rangle$.

\subsection{2 - Determine local quantization axis}
At the atom's location, determine the direction of the magnetic field vector,

\begin{equation}
    \hat{B} = \frac{\vec{B}}{|\vec{B}|}
\end{equation}

This establishes the quantization axis that the beam vectors must be compared against.

\subsection{3 - Calculate transition components}
Now that we have the quantization axis, project the beam polarizations onto the local spherical basis.  Calculate the fractions of intensity for each driving transition, P(+1), P(0), P(-1).  

For each beam $j$ with complex laboratory-frame polarization vector $\epsilon_j$ the local spherical basis should be constructed relative to $\hat{B}$.  The transition driving probabilities can be expressed as

\begin{equation}
    P_{j,q} = |e_q^*(\hat{b})\cdot \epsilon_j|^2
\end{equation}

where $q=-1,0,+1$.

Verify that $P(+1) + P(0) + P(-1) = 1$.

\subsection{4 - Track beam characteristics}
Each beam should possess a data structure containing:
\begin{itemize}
    \item Propagation Vector $\vec{k_i}$.  This says in what direction the beam is propagating.
    \item Optical Angular Frequency $\omega_i$.  Inherent frequency of the laser.
    \item Local Intensity $I_i(\vec{r})$.  Should be practically uniform due to the large Rayleigh range, but still keep track.
    \item Polarization Vector $\epsilon_i$.  Rather than saying "RCP" or "LCP", it is good to explicitly define the polarization vector in laboratory coordinates.
\end{itemize}

Remember, the wavevector of a retroreflected beam is $\vec{k}_{retro} = -\vec{k}_{incident}$.

Depending on the atom's velocity, calculate the Doppler-shifted beam frequency.

\subsection{5 - Scattering Rate and Force}
Calculate the scattering rates from each beam, for each transition.  Convert that rate into a force and apply the absorption recoil and spontaneous decay/emission recoil.

The scattering rate $R_j$ is given by

\begin{equation}
    R_j = \sum_e R_{j, g\rightarrow e}
\end{equation}

So, the absorption beams are sampled with probabilities

\begin{equation}
    P(j|scatter) = \frac{R_j}{R_{total}}
\end{equation}

\subsection{6 - Update Internal and External States}
Obviously the position and velocity will change based on these actions.  Account for this as usual.  But the internal state will also change upon absorption and emission.  Update the $F$ value and $m_F$ value.  Temporarily store the excited state quantum numbers, but retain the numbers of the state it decays down to, to be prepared for the next timestep iteration.

The spontaneous decay destination should be selected based on branching probabilities determined by the dipole matrix elements

\subsection{ChatGPT Review}
ChatGPT provides a rather nice concise overview of this flow:

\begin{verbatim}
Given atom position r, velocity v, and ground state (F, mF)

1. Compute magnetic field:
       B_vec = anti_helmholtz_field(r)
       B_mag = norm(B_vec)
       b_hat = local_quantization_axis(B_vec)

2. For each beam j:
       compute local intensity I_j(r)
       retrieve k_j, omega_j, lab polarization epsilon_j

       decompose epsilon_j into local:
           sigma_plus weight
           pi weight
           sigma_minus weight

3. For each allowed excited hyperfine manifold F':
       for q in {-1, 0, +1}:
           mF_prime = mF + q

           if mF_prime is allowed:
               compute transition strength
               compute Zeeman transition shift
               compute Doppler shift
               compute effective detuning
               compute scattering rate R[j, F', mF_prime]

4. Sum all transition rates:
       R_total = sum(all R)

5. Determine whether a scattering event occurs:
       P_scatter = 1 - exp(-R_total * dt)

       draw uniform random number u

6. If u < P_scatter:
       choose beam and excited state weighted by R

       apply absorption recoil:
           p += hbar * k_j

       choose spontaneous decay ground state
           using branching ratios

       apply spontaneous-emission recoil:
           p -= hbar * k_emit * random_direction

       update atom state:
           (F, mF) = selected decay state

7. Apply any non-scattering forces:
       gravity
       dipole force, later
       etc.

8. Update velocity and position.
\end{verbatim}

\subsection{Additional Notes}
Remember to be consistent with units throughout.

The effective detuning should be calculated as:

\begin{equation}
    \Delta_{j, g\rightarrow e}(\vec{r},\vec{v}) = \omega_j - \omega_{FF'}^{(0)} -\vec{k}_j \cdot \vec{v} - \frac{\mu_B B(\vec{r})}{\hbar}(g_F' m_F' - g_F m_F)
\end{equation}

Here, g denotes the ground state $(F, m_F)$ and e denotes the excited state $(F', m_F')$.

At the radial center where $B=0$, if this is evaluated at any timestep, just use the quantization axis from the previous timestep.  To avoid singularities or bugs.
