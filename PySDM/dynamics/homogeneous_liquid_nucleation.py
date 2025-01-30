from PySDM.dynamics.impl import register_dynamic
from PySDM.dynamics.impl import SuperParticleSpawningDynamic


@register_dynamic()
class HomogeneousLiquidNucleation(SuperParticleSpawningDynamic):
    def __init__(self):
        self.particulator = None
        self.formulae = None
        self.index = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.formulae = builder.formulae
        self.index = self.particulator.Index.identity_index(1)

    def __call__(self):
        env = {
            k: self.particulator.environment[k].to_ndarray()[0] for k in ("T", "RH")
        }  # TODO: >0D
        e_s = self.formulae.saturation_vapour_pressure.pvs_water(
            env["T"]
        )  # @Clare - is this change OK?
        j = self.formulae.homogeneous_liquid_nucleation_rate.j_liq_homo(
            env["T"], env["RH"], e_s
        )

        # TODO: take care of cases where round yields zero -> MC sampling?
        new_sd_multiplicity = round(j * env.dv * self.particulator.dt)

        if new_sd_multiplicity > 0:
            r_wet = self.formulae.r_liq_homo(env["T"], env["RH"], e_s)
            v_wet = self.formulae.trivia.volume(radius=r_wet)
            new_sd_extensive_attributes = {
                "water mass": v_wet * self.formulae.constants.rho_w,
                "dry volume": 0,
                "kappa times dry volume": 0,
            }
            self.particulator.spawn(
                seeded_particle_index=self.index,
                number_of_super_particles_to_inject=1,
                seeded_particle_multiplicity=new_sd_multiplicity,
                seeded_particle_extensive_attributes=new_sd_extensive_attributes,
            )
            # TODO: subtract the water mass from ambient vapour
