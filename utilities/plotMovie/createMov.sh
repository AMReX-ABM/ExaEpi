#!/bin/bash
module load python

GROUP="FIPS"
ZOOM="NO"
BAYAREA="NO"
for i in "$@"; do
  case $i in
    -c=*|--censusMap=*)
      CENSUS="${i#*=}"
      shift # past argument=value
      ;;
    -i=*|--infectionCases=*)
      INFECTION="${i#*=}"
      shift # past argument=value
      ;;
    -m=*|--plotMode=*) #CUML/DAILY
      OUTPUT_MODE="${i#*=}"
      shift # past argument=value
      ;;
    -n=*|--normalize=*) #RAW (no normalization), POP (normalize to population)
      NORM="${i#*=}"
      shift # past argument=value
      ;;
    -g=*|--group=*) #group data to a higher level (FIPS and GROUP are 2 options for now)
      GROUP="${i#*=}"
      shift # past argument=value
      ;;
    -z=*|--zoomin=*) #zoom in towards a location, format: "startTime, stopTime, target x, target y, zoomFactor"
      ZOOM="${i#*=}"
      shift # past argument=value
      ;;
    -b=*|--bayAreaDataFormat=*)
      BAYAREA="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

echo "PATH TO INFECTION CASES: ${INFECTION}"
echo "PATH TO CENSUS MAP: ${CENSUS}"
echo "MODE TO PLOT: ${OUTPUT_MODE}"
echo "NORMALIZE TO: ${NORM}"
echo "GROUP TO (DEFAULT: FIPS): ${GROUP}"
echo "ZOOM IN (DEFAULT: NO): ${ZOOM}"
echo "BAY AREA DATA FORMAT (DEFAULT: NO): ${BAYAREA}"

#plot everyday's cases on a cencus map
python3 plotCases.py ${INFECTION} ${CENSUS} ${OUTPUT_MODE} ${NORM} ${GROUP} ${ZOOM} ${BAYAREA}

#create movie
ffmpeg -framerate 1 -i "casefile%05d.png" -c:v libx264 -r 30 -pix_fmt yuv420p movie.mp4

#remove temporary files
rm *.png
