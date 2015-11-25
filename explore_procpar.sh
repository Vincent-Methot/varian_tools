if [ "$1" != "" ]; then
	for I in `find -name procpar | sort | grep img | grep $1`; 
	do 
		echo $I; 
		print_procpar.py $I; 
		# sleep 1; 
		echo; 
	done
else
	for I in `find -name procpar | sort | grep img`; 
	do 
		echo $I; 
		print_procpar.py $I; 
		# sleep 1; 
		echo; 
	done
fi

