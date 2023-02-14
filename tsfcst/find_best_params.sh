while getopts t:m:cv:nt:rc:aod: flag
do
    case "${flag}" in
        t) target_name=${OPTARG};;
        m) model=${OPTARG};;
        cv) cv_args=${OPTARG};;
        nt) n_trials=${OPTARG};;
        rc) reg_coef=${OPTARG};;
        aod) asofdate=${OPTARG};;
    esac
done

echo "running tsfcst.find_best_params.py in parallel with parameters:";
echo "target_name: $target_name";
echo "model: $model";
echo "cv_args: $cv_args";
echo "n_trials: $n_trials";
echo "reg_coef: $reg_coef";
echo "asofdate: asofdate";

python -m tsfcst.find_best_params -t "$target_name" -m "$model" -cv "$cv_args" -nt "$n_trials" -rc "$reg_coef" -aod "$asofdate" -p 1 -np 8 \
& python -m tsfcst.find_best_params -t "$target_name" -m "$model" -cv "$cv_args" -nt "$n_trials" -rc "$reg_coef" -aod "$asofdate" -p 2 -np 8 \
& python -m tsfcst.find_best_params -t "$target_name" -m "$model" -cv "$cv_args" -nt "$n_trials" -rc "$reg_coef" -aod "$asofdate" -p 3 -np 8 \
& python -m tsfcst.find_best_params -t "$target_name" -m "$model" -cv "$cv_args" -nt "$n_trials" -rc "$reg_coef" -aod "$asofdate" -p 4 -np 8 \
& python -m tsfcst.find_best_params -t "$target_name" -m "$model" -cv "$cv_args" -nt "$n_trials" -rc "$reg_coef" -aod "$asofdate" -p 5 -np 8 \
& python -m tsfcst.find_best_params -t "$target_name" -m "$model" -cv "$cv_args" -nt "$n_trials" -rc "$reg_coef" -aod "$asofdate" -p 6 -np 8 \
& python -m tsfcst.find_best_params -t "$target_name" -m "$model" -cv "$cv_args" -nt "$n_trials" -rc "$reg_coef" -aod "$asofdate" -p 7 -np 8 \
& python -m tsfcst.find_best_params -t "$target_name" -m "$model" -cv "$cv_args" -nt "$n_trials" -rc "$reg_coef" -aod "$asofdate" -p 8 -np 8


# examples how to run
# ./tsfcst/find_best_params.sh -t microbusiness_density -m ma -cv test -nt 100 -rc 0 -aod 2022-07-01
