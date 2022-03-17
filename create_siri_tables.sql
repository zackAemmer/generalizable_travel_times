CREATE TABLE active_trips_norway (
	datedvehiclejourney varchar,
	dataframe varchar,
	vehicle varchar,
	mode varchar,
	line varchar,
	linename varchar,
	direction varchar,
	operator varchar,
	datasource varchar,
	lat float,
	lon float,
	bearing varchar,
	delay varchar,
	nextstop varchar,
	locationtime integer,
	collectedtime integer
);
CREATE INDEX tripid_norway_idx ON active_trips_norway (datedvehiclejourney);
CREATE INDEX loctime_norway_idx ON active_trips_norway (locationtime);