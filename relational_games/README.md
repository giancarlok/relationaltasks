The data is available at https://console.cloud.google.com/storage/browser/relations-game-datasets;tab=objects

And should be downloaded into the `npz_files folder.

In order to generate the "leftof" datasets, first make sure to have downloaded the three files '`same_hexos.npz`', '`same_pentos.npz`' and '`same_stripes.npz`' files from the above link. Then run the 'generate_leftof.py' python script in the tasks folder.

To run the '`between`' task with the CoRelNet model for 2500 iterations, simply run 

`python3 ./main.py --iterations 2500 --task_name between --run 0 --model_name CoRelNet`
