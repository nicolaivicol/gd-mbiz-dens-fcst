while getopts ":t:m:c:n:r:a:s:i:" flag
do
    case "${flag}" in
        t) targetname=${OPTARG};;
        m) model=${OPTARG};;
        c) cvargs=${OPTARG};;
        s) searchargs=${OPTARG};;
        n) ntrials=${OPTARG};;
        r) regcoef=${OPTARG};;
        a) asofdate=${OPTARG};;
        i) idcol=${OPTARG};;
    esac
done

echo "running tsfcst.find_best_params.py in parallel with parameters:";
echo "targetname: $targetname";
echo "model: $model";
echo "cvargs: $cvargs";
echo "searchargs: $searchargs";
echo "ntrials: $ntrials";
echo "regcoef: $regcoef";
echo "asofdate: $asofdate";
echo "idcol: $idcol";

python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -s "$searchargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -i "$idcol" -p 1 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -s "$searchargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -i "$idcol" -p 2 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -s "$searchargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -i "$idcol" -p 3 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -s "$searchargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -i "$idcol" -p 4 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -s "$searchargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -i "$idcol" -p 5 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -s "$searchargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -i "$idcol" -p 6 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -s "$searchargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -i "$idcol" -p 7 -x 8 \
& python -m tsfcst.find_best_params -t "$targetname" -m "$model" -c "$cvargs" -s "$searchargs" -n "$ntrials" -r "$regcoef" -a "$asofdate" -i "$idcol" -p 8 -x 8

# examples how to run
# ./tsfcst/find_best_params.sh -t active -m theta -c test -s tld -n 50 -r 0.02 -a 2022-12-01 -i cfips
