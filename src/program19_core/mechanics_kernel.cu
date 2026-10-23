// Grid-stride overdamped step for a planar platelet-fibrin layer.
// Force laws match program19_core for paper_consistent and codeocean_literal.
// Thermal increments use Philox and Box-Muller, not the NumPy bit generator.

constexpr int BLOCK = 256;
constexpr double INF_DISTANCE = 1.0e300;

__device__ double pow6(double q) {
    double q2 = q * q;
    return q2 * q2 * q2;
}

__device__ double wlc_magnitude(
    double distance, double reference, double prefactor, double contour_c, double gamma_max
) {
    double safe_reference = fmax(reference, 1e-12);
    double gamma = fmin((distance - reference) / safe_reference, gamma_max);
    double normalized = gamma / contour_c;
    double denominator = fmax(1.0 - normalized, 1e-12);
    return prefactor * (0.25 / (denominator * denominator) - 0.25 + normalized);
}

__device__ double paper_stiffness(
    double mean_strain, double prefactor, double contour_c, double gamma_max
) {
    double gamma = fmin(mean_strain, gamma_max);
    double denominator = fmax(contour_c - gamma, 1e-12);
    return prefactor * (
        (contour_c * contour_c) / (2.0 * denominator * denominator * denominator) + 1.0 / contour_c
    );
}

__device__ double archive_stiffness(
    double mean_abs_strain, double prefactor, double contour_c, double gamma_max
) {
    double gamma = fmin(fabs(mean_abs_strain), gamma_max);
    double denominator = fmax(contour_c - gamma, 1e-12);
    return prefactor * (1.0 / contour_c + (contour_c * contour_c) / (2.0 * denominator));
}

__device__ void philox4x32(unsigned int c[4], unsigned long long seed) {
    unsigned int k0 = (unsigned int)seed;
    unsigned int k1 = (unsigned int)(seed >> 32);
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        unsigned long long p0 = (unsigned long long)0xD2511F53u * c[0];
        unsigned long long p1 = (unsigned long long)0xCD9E8D57u * c[2];
        unsigned int hi0 = (unsigned int)(p0 >> 32);
        unsigned int lo0 = (unsigned int)p0;
        unsigned int hi1 = (unsigned int)(p1 >> 32);
        unsigned int lo1 = (unsigned int)p1;
        c[0] = hi1 ^ c[1] ^ k0;
        c[1] = lo1;
        c[2] = hi0 ^ c[3] ^ k1;
        c[3] = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
}

__device__ void unit_gaussians(
    unsigned long long seed, unsigned long long step, unsigned int particle, unsigned int kind,
    double &gx, double &gy
) {
    unsigned int counter[4] = {
        (unsigned int)step,
        (unsigned int)(step >> 32),
        particle,
        kind
    };
    philox4x32(counter, seed);
    double u1 = ((counter[0] >> 8) + 0.5) * (1.0 / 16777216.0);
    double u2 = ((counter[1] >> 8) + 0.5) * (1.0 / 16777216.0);
    double radius = sqrt(-2.0 * log(u1));
    double angle = 6.28318530717958647692 * u2;
    gx = radius * cos(angle);
    gy = radius * sin(angle);
}

__device__ void grid_sync(int *count, int *sense) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        int generation = atomicAdd(sense, 0);
        int ticket = atomicAdd(count, 1);
        if (ticket == gridDim.x - 1) {
            atomicExch(count, 0);
            __threadfence();
            atomicAdd(sense, 1);
        } else {
            while (atomicAdd(sense, 0) == generation) {
            }
        }
    }
    __syncthreads();
}

__device__ double block_sum(double value, double *red) {
    const int tid = threadIdx.x;
    red[tid] = value;
    __syncthreads();
    for (int offset = BLOCK / 2; offset > 0; offset >>= 1) {
        if (tid < offset) red[tid] += red[tid + offset];
        __syncthreads();
    }
    double out = red[0];
    __syncthreads();
    return out;
}

__device__ double block_max(double value, double *red) {
    const int tid = threadIdx.x;
    red[tid] = value;
    __syncthreads();
    for (int offset = BLOCK / 2; offset > 0; offset >>= 1) {
        if (tid < offset) red[tid] = fmax(red[tid], red[tid + offset]);
        __syncthreads();
    }
    double out = red[0];
    __syncthreads();
    return out;
}


extern "C" __global__ void update_filopodia(
    const double *node_x, const double *node_y,
    const double *platelet_x, const double *platelet_y,
    int *filopodia, const int *arm_counts,
    int *occ_platelet, int *occ_arm, int *candidates,
    int n_nodes, int n_platelets, int max_arms, int exclusivity,
    double filo_inner, double filo_outer, int candidate_stride
) {
    if (n_platelets == 0 || max_arms == 0) return;
    double outer2 = filo_outer * filo_outer;
    int limit = candidate_stride - 1;
    for (int p = threadIdx.x; p < n_platelets; p += blockDim.x) {
        double cx = platelet_x[p];
        double cy = platelet_y[p];
        int arms = arm_counts[p];
        int base = p * max_arms;
        for (int arm = 0; arm < arms; ++arm) {
            int node = filopodia[base + arm];
            if (node < 0) continue;
            double distance = hypot(node_x[node] - cx, node_y[node] - cy);
            if (distance < filo_inner || distance > filo_outer) {
                filopodia[base + arm] = -1;
                occ_platelet[node] = -1;
                occ_arm[node] = -1;
            }
        }
        int free_arms = 0;
        for (int arm = 0; arm < arms; ++arm) {
            if (filopodia[base + arm] < 0) free_arms += 1;
        }
        int slot = p * candidate_stride;
        if (free_arms == 0) {
            candidates[slot + limit] = -1;
            continue;
        }
        int count = 0;
        for (int node = 0; node < n_nodes && count < limit; ++node) {
            double dx = node_x[node] - cx;
            double dy = node_y[node] - cy;
            if (dx * dx + dy * dy <= outer2) candidates[slot + count++] = node;
        }
        candidates[slot + limit] = count;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int p = 0; p < n_platelets; ++p) {
        int slot = p * candidate_stride;
        int count = candidates[slot + limit];
        if (count < 0) continue;
        double cx = platelet_x[p];
        double cy = platelet_y[p];
        int arms = arm_counts[p];
        int base = p * max_arms;
        int free_arms = 0;
        for (int arm = 0; arm < arms; ++arm) {
            if (filopodia[base + arm] < 0) free_arms += 1;
        }
        bool truncated = count == limit;
        for (int arm = 0; arm < arms && free_arms > 0; ++arm) {
            if (filopodia[base + arm] >= 0) continue;
            int best = -1;
            double best_distance = INF_DISTANCE;
            int scan = truncated ? n_nodes : count;
            for (int c = 0; c < scan; ++c) {
                int node = truncated ? c : candidates[slot + c];
                double distance = hypot(node_x[node] - cx, node_y[node] - cy);
                if (distance < filo_inner || distance > filo_outer) continue;
                bool used = false;
                for (int other = 0; other < arms; ++other) {
                    if (filopodia[base + other] == node) used = true;
                }
                if (used || (exclusivity && occ_platelet[node] >= 0)) continue;
                if (distance < best_distance || (distance == best_distance && (best < 0 || node < best))) {
                    best = node;
                    best_distance = distance;
                }
            }
            if (best >= 0) {
                filopodia[base + arm] = best;
                if (exclusivity) {
                    occ_platelet[best] = p;
                    occ_arm[best] = arm;
                }
            }
            free_arms -= 1;
        }
    }
}

extern "C" __global__ void integrate_chunk(
    double *node_x, double *node_y,
    double *platelet_x, double *platelet_y,
    const int *backbone_pairs, const double *backbone_l0,
    const int *cohesion_pairs, const double *cohesion_l0,
    const int *bend_i, const int *bend_j, const int *bend_k,
    const double *bend_phi0, const double *bend_dual,
    const int *incident_offsets, const int *incident_edges,
    const int *fixed_nodes,
    int *filopodia, const int *arm_counts,
    int *occ_platelet, int *occ_arm,
    double *wlc_x, double *wlc_y,
    double *bend_x, double *bend_y,
    double *contact_nx, double *contact_ny,
    double *contact_px, double *contact_py,
    double *move_nx, double *move_ny,
    double *move_px, double *move_py,
    int *candidates,
    int *error_flag,
    int n_nodes, int n_platelets, int n_backbone, int n_cohesion, int n_bending, int max_arms,
    int equation_codeocean, int exclusivity, int divide_by_attached,
    int brownian, int adaptive, int remove_com,
    double prefactor, double contour_c, double gamma_max, double ei, double archive_bend,
    double fibrin_inner, double fibrin_outer, double fibrin_force,
    double filo_inner, double filo_outer,
    double filo_f0, double filo_a, double filo_b, double filo_c,
    double sigma, double pair_cutoff, double pair_param, int lj_codeocean,
    double node_drag, double platelet_drag, double node_d, double platelet_d,
    double target_disp, double abort_disp,
    double interval_s, double remaining_in,
    int max_chunk, int substep_origin, int max_substeps,
    unsigned long long seed,
    int *barrier_count, int *barrier_sense, int *g_status,
    double *partials, double *scratch,
    double *result
) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * BLOCK + tid;
    const int stride = gridDim.x * BLOCK;
    const bool leader = blockIdx.x == 0 && tid == 0;
    __shared__ double red[BLOCK];
    (void)max_chunk;
    (void)max_substeps;

    if (gid == 0) error_flag[0] = 0;
    grid_sync(barrier_count, barrier_sense);
    if (g_status[0] != 0 || remaining_in <= fmax(1e-15, interval_s * 1e-12)) {
        if (leader) {
            result[0] = 0.0;
            result[1] = (double)g_status[0];
            result[2] = remaining_in;
            for (int i = 3; i < 17; ++i) result[i] = 0.0;
            result[15] = INF_DISTANCE;
        }
        return;
    }

    for (int i = gid; i < n_nodes; i += stride) {
        wlc_x[i] = 0.0;
        wlc_y[i] = 0.0;
        bend_x[i] = 0.0;
        bend_y[i] = 0.0;
        contact_nx[i] = 0.0;
        contact_ny[i] = 0.0;
    }
    for (int i = gid; i < n_platelets; i += stride) {
        contact_px[i] = 0.0;
        contact_py[i] = 0.0;
    }
    grid_sync(barrier_count, barrier_sense);

    double local_strain = 0.0;
    for (int edge = gid; edge < n_backbone; edge += stride) {
        int a = backbone_pairs[2 * edge];
        int b = backbone_pairs[2 * edge + 1];
        double dx = node_x[b] - node_x[a];
        double dy = node_y[b] - node_y[a];
        double distance = hypot(dx, dy);
        double safe_reference = fmax(backbone_l0[edge], 1e-12);
        local_strain = fmax(local_strain, fabs(fmin((distance - backbone_l0[edge]) / safe_reference, gamma_max)));
        double magnitude = wlc_magnitude(distance, backbone_l0[edge], prefactor, contour_c, gamma_max);
        double safe = fmax(distance, 1e-12);
        double fx = magnitude * dx / safe;
        double fy = magnitude * dy / safe;
        atomicAdd(wlc_x + a, fx);
        atomicAdd(wlc_y + a, fy);
        atomicAdd(wlc_x + b, -fx);
        atomicAdd(wlc_y + b, -fy);
    }
    for (int edge = gid; edge < n_cohesion; edge += stride) {
        int a = cohesion_pairs[2 * edge];
        int b = cohesion_pairs[2 * edge + 1];
        double dx = node_x[b] - node_x[a];
        double dy = node_y[b] - node_y[a];
        double distance = hypot(dx, dy);
        double magnitude = wlc_magnitude(distance, cohesion_l0[edge], prefactor, contour_c, gamma_max);
        double safe = fmax(distance, 1e-12);
        double fx = magnitude * dx / safe;
        double fy = magnitude * dy / safe;
        atomicAdd(wlc_x + a, fx);
        atomicAdd(wlc_y + a, fy);
        atomicAdd(wlc_x + b, -fx);
        atomicAdd(wlc_y + b, -fy);
    }
    for (int tri = gid; tri < n_bending; tri += stride) {
        int i = bend_i[tri];
        int j = bend_j[tri];
        int k = bend_k[tri];
        double ux = node_x[j] - node_x[i];
        double uy = node_y[j] - node_y[i];
        double vx = node_x[k] - node_x[j];
        double vy = node_y[k] - node_y[j];
        double lu2 = ux * ux + uy * uy;
        double lv2 = vx * vx + vy * vy;
        if (lu2 <= 1e-18 || lv2 <= 1e-18) continue;
        double phi = atan2(ux * vy - uy * vx, ux * vx + uy * vy);
        double delta = atan2(sin(phi - bend_phi0[tri]), cos(phi - bend_phi0[tri]));
        double fix, fiy, fjx, fjy, fkx, fky;
        if (equation_codeocean) {
            double lu = sqrt(lu2);
            double lv = sqrt(lv2);
            double magnitude = archive_bend * delta;
            fix = magnitude * (uy / lu);
            fiy = magnitude * (-ux / lu);
            fkx = magnitude * (vy / lv);
            fky = magnitude * (-vx / lv);
            fjx = -fix - fkx;
            fjy = -fiy - fky;
        } else {
            double stiffness = ei / fmax(bend_dual[tri], 1e-12);
            double dux = stiffness * delta * (uy / lu2);
            double duy = stiffness * delta * (-ux / lu2);
            double dvx = stiffness * delta * (-vy / lv2);
            double dvy = stiffness * delta * (vx / lv2);
            fix = dux;
            fiy = duy;
            fjx = -dux + dvx;
            fjy = -duy + dvy;
            fkx = -dvx;
            fky = -dvy;
        }
        atomicAdd(bend_x + i, fix);
        atomicAdd(bend_y + i, fiy);
        atomicAdd(bend_x + j, fjx);
        atomicAdd(bend_y + j, fjy);
        atomicAdd(bend_x + k, fkx);
        atomicAdd(bend_y + k, fky);
    }

    int pair_count = n_platelets * n_nodes;
    double outer2 = fibrin_outer * fibrin_outer;
    for (int index = gid; index < pair_count; index += stride) {
        int p = index / n_nodes;
        int node = index - p * n_nodes;
        double dx = node_x[node] - platelet_x[p];
        double dy = node_y[node] - platelet_y[p];
        double dist2 = dx * dx + dy * dy;
        if (dist2 > outer2) continue;
        double distance = sqrt(dist2);
        if (distance <= 1e-12) {
            atomicExch(error_flag, 5);
            continue;
        }
        double magnitude = distance < fibrin_inner ? fibrin_force : -fibrin_force;
        double fx = magnitude * dx / distance;
        double fy = magnitude * dy / distance;
        atomicAdd(contact_nx + node, fx);
        atomicAdd(contact_ny + node, fy);
        atomicAdd(contact_px + p, -fx);
        atomicAdd(contact_py + p, -fy);
    }
    grid_sync(barrier_count, barrier_sense);

    __syncthreads();
    int overlaps_now = 0;
    double min_now = INF_DISTANCE;
    if (leader && n_platelets >= 2) {
        for (int a = 0; a < n_platelets; ++a) {
            for (int b = a + 1; b < n_platelets; ++b) {
                double dx = platelet_x[b] - platelet_x[a];
                double dy = platelet_y[b] - platelet_y[a];
                double distance = hypot(dx, dy);
                if (distance < min_now) min_now = distance;
                if (distance > pair_cutoff) continue;
                if (distance <= 1e-9) {
                    atomicExch(error_flag, 5);
                    continue;
                }
                if (distance < pair_cutoff) overlaps_now += 1;
                double q = sigma / distance;
                double q6 = pow6(q);
                double q12 = q6 * q6;
                double scalar = lj_codeocean
                    ? 12.0 * pair_param * pair_param * (q12 - q6) / distance
                    : 24.0 * pair_param * (2.0 * q12 - q6) / distance;
                double fx = -scalar * dx / distance;
                double fy = -scalar * dy / distance;
                contact_px[a] += fx;
                contact_py[a] += fy;
                contact_px[b] -= fx;
                contact_py[b] -= fy;
            }
        }
    }
    __syncthreads();
    grid_sync(barrier_count, barrier_sense);

    if (leader && error_flag[0]) g_status[0] = error_flag[0];
    grid_sync(barrier_count, barrier_sense);
    int failed = g_status[0];

    int local_attached = 0;
    if (!failed && max_arms > 0) {
        for (int p = gid; p < n_platelets; p += stride) {
            double cx = platelet_x[p];
            double cy = platelet_y[p];
            int arms = arm_counts[p];
            int base = p * max_arms;
            int attached_nodes = 0;
            for (int arm = 0; arm < arms; ++arm) {
                if (filopodia[base + arm] >= 0) attached_nodes += 1;
            }
            double divisor = divide_by_attached ? (double)max(attached_nodes, 1) : 1.0;
            double px = 0.0;
            double py = 0.0;
            for (int arm = 0; arm < arms; ++arm) {
                int node = filopodia[base + arm];
                if (node < 0) continue;
                int begin = incident_offsets[node];
                int end = incident_offsets[node + 1];
                int incident = end - begin;
                double stiffness = 0.0;
                if (incident > 0) {
                    if (equation_codeocean) {
                        double acc = 0.0;
                        for (int e = begin; e < end; ++e) {
                            int edge = incident_edges[e];
                            int a = backbone_pairs[2 * edge];
                            int b = backbone_pairs[2 * edge + 1];
                            double distance = hypot(node_x[b] - node_x[a], node_y[b] - node_y[a]);
                            double reference = fmax(backbone_l0[edge], 1e-12);
                            acc += fabs((distance - backbone_l0[edge]) / reference);
                        }
                        stiffness = archive_stiffness(acc / (double)incident, prefactor, contour_c, gamma_max);
                    } else {
                        double acc = 0.0;
                        for (int e = begin; e < end; ++e) {
                            int edge = incident_edges[e];
                            int a = backbone_pairs[2 * edge];
                            int b = backbone_pairs[2 * edge + 1];
                            double distance = hypot(node_x[b] - node_x[a], node_y[b] - node_y[a]);
                            double reference = fmax(backbone_l0[edge], 1e-12);
                            acc += (distance - backbone_l0[edge]) / reference;
                        }
                        stiffness = paper_stiffness(acc / (double)incident, prefactor, contour_c, gamma_max);
                    }
                }
                double denominator = filo_b * stiffness + filo_c;
                double response = fabs(denominator) <= 1e-12 ? 0.0 : filo_a * stiffness / denominator;
                double magnitude = fmax(0.0, 0.5 * (filo_f0 + response)) / divisor;
                double dx = cx - node_x[node];
                double dy = cy - node_y[node];
                double distance = hypot(dx, dy);
                if (distance <= 1e-12) continue;
                double fx = magnitude * dx / distance;
                double fy = magnitude * dy / distance;
                atomicAdd(contact_nx + node, fx);
                atomicAdd(contact_ny + node, fy);
                px -= fx;
                py -= fy;
                local_attached += 1;
            }
            contact_px[p] += px;
            contact_py[p] += py;
        }
    }
    grid_sync(barrier_count, barrier_sense);

    double local_node_force = 0.0;
    double local_platelet_force = 0.0;
    double local_wlc_sq = 0.0;
    double local_bend_sq = 0.0;
    double local_contact_sq = 0.0;
    double local_net_x = 0.0;
    double local_net_y = 0.0;
    for (int i = gid; i < n_nodes; i += stride) {
        double wx = wlc_x[i], wy = wlc_y[i];
        double bx = bend_x[i], by = bend_y[i];
        double cx = contact_nx[i], cy = contact_ny[i];
        local_node_force = fmax(local_node_force, hypot(wx + bx + cx, wy + by + cy));
        local_wlc_sq += wx * wx + wy * wy;
        local_bend_sq += bx * bx + by * by;
        local_contact_sq += cx * cx + cy * cy;
        local_net_x += wx + bx + cx;
        local_net_y += wy + by + cy;
    }
    for (int i = gid; i < n_platelets; i += stride) {
        local_platelet_force = fmax(local_platelet_force, hypot(contact_px[i], contact_py[i]));
        local_contact_sq += contact_px[i] * contact_px[i] + contact_py[i] * contact_py[i];
        local_net_x += contact_px[i];
        local_net_y += contact_py[i];
    }
    const int pitch = 8;
    double b_node_force = block_max(local_node_force, red);
    double b_platelet_force = block_max(local_platelet_force, red);
    double b_wlc = block_sum(local_wlc_sq, red);
    double b_bend = block_sum(local_bend_sq, red);
    double b_contact = block_sum(local_contact_sq, red);
    double b_net_x = block_sum(local_net_x, red);
    double b_net_y = block_sum(local_net_y, red);
    double b_strain = block_max(local_strain, red);
    double b_attached = block_sum((double)local_attached, red);
    if (tid == 0) {
        int slot = blockIdx.x;
        partials[slot] = b_node_force;
        partials[pitch + slot] = b_platelet_force;
        partials[2 * pitch + slot] = b_wlc;
        partials[3 * pitch + slot] = b_bend;
        partials[4 * pitch + slot] = b_contact;
        partials[5 * pitch + slot] = b_net_x;
        partials[6 * pitch + slot] = b_net_y;
        partials[7 * pitch + slot] = b_strain;
        partials[8 * pitch + slot] = b_attached;
    }
    grid_sync(barrier_count, barrier_sense);
    int n_objects = n_nodes + n_platelets;
    double max_node_force = 0.0;
    double max_platelet_force = 0.0;
    double wlc_now = 0.0;
    double bend_now = 0.0;
    double contact_now = 0.0;
    double max_strain = 0.0;
    double net_x = 0.0;
    double net_y = 0.0;
    int attached_now = 0;
    if (leader) {
        max_node_force = partials[0];
        max_platelet_force = partials[pitch];
        double wlc_sq = 0.0, bend_sq = 0.0, contact_sq = 0.0, attached_sum = 0.0;
        max_strain = partials[7 * pitch];
        for (int b = 0; b < gridDim.x; ++b) {
            max_node_force = fmax(max_node_force, partials[b]);
            max_platelet_force = fmax(max_platelet_force, partials[pitch + b]);
            wlc_sq += partials[2 * pitch + b];
            bend_sq += partials[3 * pitch + b];
            contact_sq += partials[4 * pitch + b];
            net_x += partials[5 * pitch + b];
            net_y += partials[6 * pitch + b];
            max_strain = fmax(max_strain, partials[7 * pitch + b]);
            attached_sum += partials[8 * pitch + b];
        }
        wlc_now = n_nodes > 0 ? sqrt(wlc_sq / (double)n_nodes) : 0.0;
        bend_now = n_nodes > 0 ? sqrt(bend_sq / (double)n_nodes) : 0.0;
        contact_now = n_objects > 0 ? sqrt(contact_sq / (double)n_objects) : 0.0;
        attached_now = (int)attached_sum;
        if (!failed) {
            double maximum_speed = fmax(max_node_force / node_drag, max_platelet_force / platelet_drag);
            double internal_dt = remaining_in;
            if (adaptive && maximum_speed * internal_dt > target_disp) {
                internal_dt = target_disp / maximum_speed;
            }
            if (!isfinite(internal_dt) || internal_dt <= 0.0) g_status[0] = 3;
            else scratch[0] = internal_dt;
        }
    }
    grid_sync(barrier_count, barrier_sense);
    failed = g_status[0];
    double dt = scratch[0];

    if (!failed) {
        double node_noise = brownian ? sqrt(2.0 * node_d * dt) : 0.0;
        double platelet_noise = brownian ? sqrt(2.0 * platelet_d * dt) : 0.0;
        unsigned long long step_index = (unsigned long long)substep_origin;
        for (int i = gid; i < n_nodes; i += stride) {
            move_nx[i] = dt * (wlc_x[i] + bend_x[i] + contact_nx[i]) / node_drag;
            move_ny[i] = dt * (wlc_y[i] + bend_y[i] + contact_ny[i]) / node_drag;
            if (brownian) {
                double gx, gy;
                unit_gaussians(seed, step_index, (unsigned int)i, 1u, gx, gy);
                move_nx[i] += node_noise * gx;
                move_ny[i] += node_noise * gy;
            }
        }
        for (int i = gid; i < n_platelets; i += stride) {
            move_px[i] = dt * contact_px[i] / platelet_drag;
            move_py[i] = dt * contact_py[i] / platelet_drag;
            if (brownian) {
                double gx, gy;
                unit_gaussians(seed, step_index, (unsigned int)i, 2u, gx, gy);
                move_px[i] += platelet_noise * gx;
                move_py[i] += platelet_noise * gy;
            }
        }
    }
    grid_sync(barrier_count, barrier_sense);

    double local_sum_x = 0.0, local_sum_y = 0.0;
    if (!failed) {
        for (int i = gid; i < n_nodes; i += stride) {
            local_sum_x += move_nx[i];
            local_sum_y += move_ny[i];
        }
        for (int i = gid; i < n_platelets; i += stride) {
            local_sum_x += move_px[i];
            local_sum_y += move_py[i];
        }
    }
    double b_sum_x = block_sum(local_sum_x, red);
    double b_sum_y = block_sum(local_sum_y, red);
    if (tid == 0) {
        partials[blockIdx.x] = b_sum_x;
        partials[pitch + blockIdx.x] = b_sum_y;
    }
    grid_sync(barrier_count, barrier_sense);
    if (leader) {
        double summed_x = 0.0, summed_y = 0.0;
        for (int b = 0; b < gridDim.x; ++b) {
            summed_x += partials[b];
            summed_y += partials[pitch + b];
        }
        scratch[2] = summed_x;
        scratch[3] = summed_y;
    }
    grid_sync(barrier_count, barrier_sense);
    double sum_x = scratch[2];
    double sum_y = scratch[3];
    if (!failed && remove_com && n_objects > 0) {
        double common_x = sum_x / (double)n_objects;
        double common_y = sum_y / (double)n_objects;
        for (int i = gid; i < n_nodes; i += stride) {
            move_nx[i] -= common_x;
            move_ny[i] -= common_y;
        }
        for (int i = gid; i < n_platelets; i += stride) {
            move_px[i] -= common_x;
            move_py[i] -= common_y;
        }
    }
    if (!failed) {
        for (int i = gid; i < n_nodes; i += stride) {
            if (fixed_nodes[i]) {
                move_nx[i] = 0.0;
                move_ny[i] = 0.0;
            }
        }
    }
    grid_sync(barrier_count, barrier_sense);

    double local_node_move = 0.0, local_platelet_move = 0.0;
    if (!failed) {
        for (int i = gid; i < n_nodes; i += stride) {
            local_node_move = fmax(local_node_move, hypot(move_nx[i], move_ny[i]));
        }
        for (int i = gid; i < n_platelets; i += stride) {
            local_platelet_move = fmax(local_platelet_move, hypot(move_px[i], move_py[i]));
        }
    }
    double b_node_move = block_max(local_node_move, red);
    double b_platelet_move = block_max(local_platelet_move, red);
    if (tid == 0) {
        partials[blockIdx.x] = b_node_move;
        partials[pitch + blockIdx.x] = b_platelet_move;
    }
    grid_sync(barrier_count, barrier_sense);
    double max_node_move = 0.0;
    double max_platelet_move = 0.0;
    if (leader) {
        max_node_move = partials[0];
        max_platelet_move = partials[pitch];
        for (int b = 1; b < gridDim.x; ++b) {
            max_node_move = fmax(max_node_move, partials[b]);
            max_platelet_move = fmax(max_platelet_move, partials[pitch + b]);
        }
    }
    if (!failed && leader) {
        double requested = fmax(max_node_move, max_platelet_move);
        if (!isfinite(requested)) g_status[0] = 3;
        else if (requested > abort_disp) {
            g_status[0] = 4;
            scratch[1] = requested;
        }
    }
    grid_sync(barrier_count, barrier_sense);
    failed = g_status[0];

    if (!failed) {
        for (int i = gid; i < n_nodes; i += stride) {
            node_x[i] += move_nx[i];
            node_y[i] += move_ny[i];
        }
        for (int i = gid; i < n_platelets; i += stride) {
            platelet_x[i] += move_px[i];
            platelet_y[i] += move_py[i];
        }
    }
    grid_sync(barrier_count, barrier_sense);
    if (leader) {
        result[0] = failed ? 0.0 : 1.0;
        result[1] = (double)failed;
        result[2] = failed ? remaining_in : fmax(0.0, remaining_in - dt);
        result[3] = failed ? 0.0 : max_node_move;
        result[4] = failed ? 0.0 : max_platelet_move;
        result[5] = wlc_now;
        result[6] = bend_now;
        result[7] = contact_now;
        result[8] = max_strain;
        result[9] = max_node_force;
        result[10] = max_platelet_force;
        result[11] = net_x;
        result[12] = net_y;
        result[13] = (double)attached_now;
        result[14] = (double)overlaps_now;
        result[15] = min_now;
        result[16] = scratch[1];
    }
}

__device__ bool shares_fiber(
    const int *fiber_ids, const int *fiber_counts, int stride, int a, int b
) {
    int count_a = fiber_counts[a];
    int count_b = fiber_counts[b];
    for (int i = 0; i < count_a; ++i) {
        int fiber = fiber_ids[a * stride + i];
        for (int j = 0; j < count_b; ++j) {
            if (fiber == fiber_ids[b * stride + j]) return true;
        }
    }
    return false;
}

extern "C" __global__ void gather_cohesion_pairs(
    const double *node_x, const double *node_y,
    int n_nodes, double radius2, int capacity,
    int *out_a, int *out_b, int *out_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    double xi = node_x[i];
    double yi = node_y[i];
    for (int j = i + 1; j < n_nodes; ++j) {
        double dx = node_x[j] - xi;
        double dy = node_y[j] - yi;
        if (dx * dx + dy * dy > radius2) continue;
        int place = atomicAdd(out_count, 1);
        if (place < capacity) {
            out_a[place] = i;
            out_b[place] = j;
        }
    }
}

extern "C" __global__ void clear_pair_bits(unsigned int *bits, int words) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) bits[i] = 0u;
}

extern "C" __global__ void set_pair_bits(
    const int *pairs, int n_pairs, int n_nodes, unsigned int *bits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    int a = pairs[2 * i];
    int b = pairs[2 * i + 1];
    if (a > b) {
        int swap = a;
        a = b;
        b = swap;
    }
    unsigned long long idx = (unsigned long long)a * (unsigned long long)n_nodes + (unsigned long long)b;
    atomicOr(bits + (idx >> 5), 1u << (unsigned)(idx & 31ull));
}

extern "C" __global__ void mark_cohesion_candidates(
    const int *cand_a, const int *cand_b, int n_candidates,
    const unsigned int *bits, int n_nodes,
    const int *fiber_ids, const int *fiber_counts, int fiber_stride,
    const int *degree, int max_total, int exclude_same_fiber,
    int *flags
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_candidates) return;
    int a = cand_a[i];
    int b = cand_b[i];
    unsigned long long idx = (unsigned long long)a * (unsigned long long)n_nodes + (unsigned long long)b;
    int keep = (bits[idx >> 5] & (1u << (unsigned)(idx & 31ull))) == 0u;
    if (keep && exclude_same_fiber && shares_fiber(fiber_ids, fiber_counts, fiber_stride, a, b)) {
        keep = 0;
    }
    if (keep && (degree[a] >= max_total || degree[b] >= max_total)) keep = 0;
    flags[i] = keep;
}

extern "C" __global__ void count_cohesion_flags(const int *flags, int n, int *out) {
    __shared__ int partial[256];
    int sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += flags[i];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = partial[0];
}

extern "C" __global__ void accept_cohesion_links(
    const double *node_x, const double *node_y,
    const int *cand_a, const int *cand_b, int n_candidates,
    int *degree, int *fresh,
    int max_total, int max_new,
    int *out_a, int *out_b, double *out_length, int *out_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int accepted = 0;
    for (int candidate = 0; candidate < n_candidates; ++candidate) {
        int a = cand_a[candidate];
        int b = cand_b[candidate];
        if (degree[a] >= max_total || degree[b] >= max_total) continue;
        if (fresh[a] >= max_new || fresh[b] >= max_new) continue;
        double dx = node_x[b] - node_x[a];
        double dy = node_y[b] - node_y[a];
        double length = hypot(dx, dy);
        if (length <= 1e-12) continue;
        out_a[accepted] = a;
        out_b[accepted] = b;
        out_length[accepted] = length;
        degree[a] += 1;
        degree[b] += 1;
        fresh[a] += 1;
        fresh[b] += 1;
        accepted += 1;
    }
    out_count[0] = accepted;
}
