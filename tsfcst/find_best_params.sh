while getopts ":t:m:c:n:r:a:" flag
do
    case "${flag}" in
        t) targetname=${OPTARG};;
        m) model=${OPTARG};;
        c) cvargs=${OPTARG};;
        n) ntrials=${OPTARG};;
        r) regcoef=${OPTARG};;
        a) asofdate=${OPTARG};;
    esac
done

echo "running tsfcst.find_best_params.py in parallel with parameters:";
echo "targetname: $targetname";
echo "model: $model";
echo "cvargs: $cvargs";
echo "ntrials: $ntrials";
echo "regcoef: $regcoef";
echo "asofdate: $asofdate";

python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -p 1 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -p 2 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -p 3 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -p 4 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -p 5 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -p 6 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -p 7 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -p 8 -x 8

# examples how to run
# ./tsfcst/find_best_params.sh -t microbusiness_density -m ma -c test -n 100 -r 0 -a 2022-07-01
