while getopts t:g:a:f:m:n:i:o:e:w:k:d: flag
do
    case "${flag}" in
        t) target_name=${OPTARG};;
        g) tag=${OPTARG};;
        a) asofdate=${OPTARG};;
        f) fromdate=${OPTARG};;
        m) method=${OPTARG};;
        n) ntrials=${OPTARG};;
        i) naive=${OPTARG};;
        o) ma=${OPTARG};;
        k) ema=${OPTARG};;
        d) driftr=${OPTARG};;
        e) theta=${OPTARG};;
        w) hw=${OPTARG};;
    esac
done

echo "running tsfcst.find_best_weights.py in parallel with parameters:";
echo "target_name: $target_name";
echo "tag: $tag";
echo "asofdate: $asofdate";
echo "fromdate: $fromdate";
echo "method: $method";
echo "ntrials: $ntrials";
echo "naive: $naive";
echo "ma: $ma";
echo "ema: $ema";
echo "driftr: $driftr";
echo "theta: $theta";
echo "hw: $hw";

python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -m "$method" -n "$ntrials" -i "$naive" -o "$ma" -k "$ema" -d "$driftr" -e "$theta" -w "$hw" -p 1 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -m "$method" -n "$ntrials" -i "$naive" -o "$ma" -k "$ema" -d "$driftr" -e "$theta" -w "$hw" -p 2 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -m "$method" -n "$ntrials" -i "$naive" -o "$ma" -k "$ema" -d "$driftr" -e "$theta" -w "$hw" -p 3 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -m "$method" -n "$ntrials" -i "$naive" -o "$ma" -k "$ema" -d "$driftr" -e "$theta" -w "$hw" -p 4 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -m "$method" -n "$ntrials" -i "$naive" -o "$ma" -k "$ema" -d "$driftr" -e "$theta" -w "$hw" -p 5 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -m "$method" -n "$ntrials" -i "$naive" -o "$ma" -k "$ema" -d "$driftr" -e "$theta" -w "$hw" -p 6 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -m "$method" -n "$ntrials" -i "$naive" -o "$ma" -k "$ema" -d "$driftr" -e "$theta" -w "$hw" -p 7 -x 8 \
& python -m tsfcst.find_best_weights -t "$target_name" -g "$tag" -a "$asofdate" -f "$fromdate" -m "$method" -n "$ntrials" -i "$naive" -o "$ma" -k "$ema" -d "$driftr" -e "$theta" -w "$hw" -p 8 -x 8
