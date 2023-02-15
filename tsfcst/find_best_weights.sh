while getopts t:g:a:f:n:i:o:e:w: flag
do
    case "${flag}" in
        t) target_name=${OPTARG};;
        g) tag=${OPTARG};;
        a) asofdate=${OPTARG};;
        f) fromdate=${OPTARG};;
        n) ntrials=${OPTARG};;
        i) naive=${OPTARG};;
        o) ma=${OPTARG};;
        e) theta=${OPTARG};;
        w) hw=${OPTARG};;
    esac
done

echo "running tsfcst.find_best_weights.py in parallel with parameters:";
echo "target_name: $target_name";
echo "tag: $tag";
echo "asofdate: $asofdate";
echo "fromdate: $fromdate";
echo "ntrials: $ntrials";
echo "naive: $naive";
echo "ma: $ma";
echo "theta: $theta";
echo "hw: $hw";

python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -n "$ntrials" -i "$naive" -o "$ma" -e "$theta" -w "$hw" -p 1 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -n "$ntrials" -i "$naive" -o "$ma" -e "$theta" -w "$hw" -p 2 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -n "$ntrials" -i "$naive" -o "$ma" -e "$theta" -w "$hw" -p 3 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -n "$ntrials" -i "$naive" -o "$ma" -e "$theta" -w "$hw" -p 4 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -n "$ntrials" -i "$naive" -o "$ma" -e "$theta" -w "$hw" -p 5 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -n "$ntrials" -i "$naive" -o "$ma" -e "$theta" -w "$hw" -p 6 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -n "$ntrials" -i "$naive" -o "$ma" -e "$theta" -w "$hw" -p 7 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -n "$ntrials" -i "$naive" -o "$ma" -e "$theta" -w "$hw" -p 8 -x 8

# example how to run
# ./tsfcst/find_best_weights.sh -t microbusiness_density -g cv -a "2022-07-01" -f none -n 200 -i "microbusiness_density-20220701-naive-test-trend_level_damp-1-0_0" -o "microbusiness_density-20220701-ma-test-trend_level_damp-100-0_0" -e "microbusiness_density-20220701-theta-test-trend_level_damp-100-0_0" -w "microbusiness_density-20220701-hw-test-trend_level_damp-100-0_0"
# ./tsfcst/find_best_weights.sh -t microbusiness_density -g test -a "2022-10-01" -f "2022-08-01" -n 100 -i "microbusiness_density-20220701-naive-test-trend_level_damp-1-0_0" -o "microbusiness_density-20220701-ma-test-trend_level_damp-100-0_0" -e "microbusiness_density-20220701-theta-test-trend_level_damp-100-0_0" -w "microbusiness_density-20220701-hw-test-trend_level_damp-100-0_0"
# ./tsfcst/find_best_weights.sh -t microbusiness_density -g test -a "2022-10-01" -f none -n 200 -i "microbusiness_density-20221001-naive-test-trend_level_damp-1-0_0" -o "microbusiness_density-20221001-ma-test-trend_level_damp-100-0_0" -e "microbusiness_density-20221001-theta-test-trend_level_damp-100-0_0" -w "microbusiness_density-20221001-hw-test-trend_level_damp-100-0_0"
