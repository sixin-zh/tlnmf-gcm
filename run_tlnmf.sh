mkdir -p ./results_pertl20/
for rid in `seq 1 5`;
do
	ws=40
	python examples/tlnmf2_sci_batch.py -sn nonstationary440_sim$rid -ws "$ws"e-3 -K 2 -toltl 1e-7 -pertl 20 -runid 1 -win 4 -epsnmf 5e-7
done
