import torch
import os
from VCFWriter import VCFWriter

d = {} #Add environment variables

GPU = 0
device = torch.device('cpu')

cvae_path = 'cvae_16.pth'

NUM_ELEM_PER_ANCESTRY = 85
ANC_MUL = 4

net = torch.load(cvae_path, map_location=device)
del net.DiscriminatorList


def sample_simulation(sim, num_elem=85):
    sim_t = sim.T
    dot_p = sim.matmul(sim_t)
    dot_p /= sim.shape[1]
    dot_p_sum = torch.sum(dot_p, dim=0) / dot_p.shape[0] * 100
    val, ind = dot_p_sum.sort()

    print(dot_p.max(), dot_p.min())
    print((dot_p == 1).sum(), (dot_p < 1).sum())

    sim = sim[ind[0:num_elem],:]
    return sim


## Simulation of new samples
_CHROMOSOME = 20
gen = 8
DATASET_ROOT = '/dataset-simulated/admixed-simulation-output-test/'
template_path = os.path.join(DATASET_ROOT, 'chm{}'.format(_CHROMOSOME), 'gen_{}'.format(gen), 'test') + '/test-admix.query.vcf'

SIMULATION_OUTPUT_PATH = os.path.join('/cvae-simulations/', 'chm{}'.format(_CHROMOSOME))

out_path_simulation = SIMULATION_OUTPUT_PATH+ '/test-write-simulation.vcf'
out_path_map = SIMULATION_OUTPUT_PATH + '/test-write-simulation.map'

if not os.path.exists(SIMULATION_OUTPUT_PATH):
    os.makedirs(SIMULATION_OUTPUT_PATH)


with torch.no_grad():
    net.eval()

    simulations_list = []
    maps_list = []
    for i in range(3):
        # Generation
        sim_all, _ = net.simulate(device=device, single_ancestry=True, batch_size=NUM_ELEM_PER_ANCESTRY*ANC_MUL, use_ancestry=i)
        sim_all = sim_all.sign()

        sim_all = sample_simulation(sim_all, num_elem=NUM_ELEM_PER_ANCESTRY*2)

        sim_a = sim_all[0:NUM_ELEM_PER_ANCESTRY,:]
        sim_b = sim_all[NUM_ELEM_PER_ANCESTRY:, :]

        sim = torch.cat([sim_a[...,None], sim_b[...,None]], dim=2)
        sim = (sim + 2)/2

        map = torch.zeros(NUM_ELEM_PER_ANCESTRY) + i
        simulations_list.append(sim.cpu())
        maps_list.append(map.cpu())

    simulation = torch.cat(simulations_list, dim=0)
    maps = torch.cat(maps_list)

simulation = simulation.cpu().numpy()
maps = maps.cpu().numpy()

print(simulation.shape, maps.shape)
BASE_HEADER = 'CVAE'
vcfw = VCFWriter()
vcfw.write_vcf(simulation,out_path_simulation,template_path=template_path,base_header=BASE_HEADER)
vcfw.write_map(maps,out_path_map,base_header=BASE_HEADER, printout=True)