mkdir -p results_dct2
for rid in `seq 1 5`;
do
        ws=40
	python examples/isnmf_dctsci_batch.py -sn nonstationary440_sim$rid -ws "$ws"e-3 -K 2 -runid 1 -win 4 -epsnmf 5e-7 -dct 2
done
