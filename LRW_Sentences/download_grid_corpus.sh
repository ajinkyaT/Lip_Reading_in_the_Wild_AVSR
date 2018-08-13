#preparing for download 
mkdir "data/gridcorpus"
cd "data/gridcorpus"
mkdir "raw" "align" "video"
cd "raw" && mkdir "align" "video"

for i in `seq $1 $2`

do
	if [ $i != 21 ]
	then
	    printf "\n\n------------------------- Downloading $i th speaker -------------------------\n\n"
	    
	    #download the video of the ith speaker
	    cd "align" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar" > "s$i.tar" && cd ..
	    cd "video" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip" && cd ..

	    if [ "$3" = "y" ]
	    then
	        unzip -q "video/s$i.zip" -d "../video"
	        mkdir "../align/s$i"
	        tar -xf "align/s$i.tar" -C "../align/s$i"
	        rm -rf "video/s$i.zip" "align/s$i.tar"
	    fi
	fi
done