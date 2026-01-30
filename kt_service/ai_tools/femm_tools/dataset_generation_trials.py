from synthetic_datasets_generator import simulate_EIT_monitoring_pyeit
from kt_service.ai_tools.mesh_tools.mesh_service_trials import *

def test_dataset_generation():
    for i, data in enumerate(get_test_data(), start=1):
        start_meshing_time = time.time()
        image, meshdata = create_mesh(['0.682', '0.682'], data,
            8,
            1.3, 0, True,
            show_meshing_result_method="no",
            is_saving_to_file=True,
            export_filename=f"dataset {i}.{get_dataset_file_format(i)}")
        end_meshing_time = time.time()
        print(f"Execution time of mesh creation in dataset {i} with mesh size parameter {8}: {end_meshing_time - start_meshing_time:.2f} seconds")
        filename = f"generation_results_{i}.dat"
        simulation_results, simulation_time = simulate_EIT_monitoring_pyeit(meshdata, N_minutes=5, isSaveToFile=True, filename=filename, materials_location=".")
        lines_count = 0
        cols_count = 0
        with open(filename, 'r') as file:
            for line in file:
                lines_count += 1
                if lines_count == 1:
                    cols_count = len(line.split(' '))
        print(f"Generated dataset matrix is {cols_count}X{lines_count} size. Generation time is {simulation_time:.2f} seconds ")

if __name__ == "__main__":
    test_dataset_generation()

