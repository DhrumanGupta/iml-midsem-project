/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import { useState } from "react";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

const SimulationInput = ({
  initialPopulation,
  setInitialPopulation,
  stateConfig,
  setStateConfig,
  schoolLockdown,
  setSchoolLockdown,
  officeLockdown,
  setOfficeLockdown,
}: {
  initialPopulation: { adults: number[]; students: number[] };
  setInitialPopulation: React.Dispatch<
    React.SetStateAction<{ adults: number[]; students: number[] }>
  >;
  stateConfig: number[];
  setStateConfig: React.Dispatch<React.SetStateAction<number[]>>;
  schoolLockdown: { days: string; intensities: string; durations: string };
  setSchoolLockdown: React.Dispatch<
    React.SetStateAction<{
      days: string;
      intensities: string;
      durations: string;
    }>
  >;
  officeLockdown: { days: string; intensities: string; durations: string };
  setOfficeLockdown: React.Dispatch<
    React.SetStateAction<{
      days: string;
      intensities: string;
      durations: string;
    }>
  >;
}) => {
  const handlePopulationChange = (
    category: "adults" | "students",
    index: number,
    value: string
  ) => {
    const numValue = parseFloat(value) || 0;
    setInitialPopulation((prev) => ({
      ...prev,
      [category]: prev[category].map((v, i) => (i === index ? numValue : v)),
    }));
  };

  const handleStateConfigChange = (index: number, value: string) => {
    const numValue = parseFloat(value) || 0;
    setStateConfig((prev) => prev.map((v, i) => (i === index ? numValue : v)));
  };

  return (
    <div className="container mx-auto p-4 space-y-6 max-h-[100vh] overflow-y-auto">
      <Card>
        <CardHeader>
          <CardTitle>Population Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Adults row */}
          <div className="space-y-2">
            <Label>Adults</Label>
            <div className="grid grid-cols-3 gap-4">
              {initialPopulation.adults.map((value, index) => (
                <div key={`adults-${index}`}>
                  <Input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={value}
                    onChange={(e) =>
                      handlePopulationChange("adults", index, e.target.value)
                    }
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Students row */}
          <div className="space-y-2">
            <Label>Students</Label>
            <div className="grid grid-cols-3 gap-4">
              {initialPopulation.students.map((value, index) => (
                <Input
                  key={`students-${index}`}
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={value}
                  onChange={(e) =>
                    handlePopulationChange("students", index, e.target.value)
                  }
                />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>State Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label htmlFor="student-percentage">Student Percentage</Label>
              <Input
                id="student-percentage"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={stateConfig[0]}
                onChange={(e) => handleStateConfigChange(0, e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="adult-percentage">Adult Percentage</Label>
              <Input
                id="adult-percentage"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={stateConfig[1]}
                onChange={(e) => handleStateConfigChange(1, e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="house-size">House Size</Label>
              <Input
                id="house-size"
                type="number"
                min="1"
                value={stateConfig[2]}
                onChange={(e) => handleStateConfigChange(2, e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="school-size">School Size</Label>
              <Input
                id="school-size"
                type="number"
                min="1"
                value={stateConfig[3]}
                onChange={(e) => handleStateConfigChange(3, e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="work-size">Work Size</Label>
              <Input
                id="work-size"
                type="number"
                min="1"
                value={stateConfig[4]}
                onChange={(e) => handleStateConfigChange(4, e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="beta">Beta</Label>
              <Input
                id="beta"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={stateConfig[5]}
                onChange={(e) => handleStateConfigChange(5, e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="gamma">Gamma</Label>
              <Input
                id="gamma"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={stateConfig[6]}
                onChange={(e) => handleStateConfigChange(6, e.target.value)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>School Lockdown Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="school-lockdown-days">School Lockdown Days </Label>
            <Input
              id="school-lockdown-days"
              type="text"
              value={schoolLockdown.days}
              onChange={(e) =>
                setSchoolLockdown((prev) => ({
                  ...prev,
                  days: e.target.value,
                }))
              }
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="school-lockdown-intensities">
              School Lockdown Intensities
            </Label>
            <Input
              id="school-lockdown-intensities"
              type="text"
              value={schoolLockdown.intensities}
              onChange={(e) =>
                setSchoolLockdown((prev) => ({
                  ...prev,
                  intensities: e.target.value,
                }))
              }
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="school-lockdown-durations">
              School Lockdown Durations
            </Label>
            <Input
              id="school-lockdown-durations"
              type="text"
              value={schoolLockdown.durations}
              onChange={(e) =>
                setSchoolLockdown((prev) => ({
                  ...prev,
                  durations: e.target.value,
                }))
              }
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Office Lockdown Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="office-lockdown-days">Office Lockdown Days</Label>
            <Input
              id="office-lockdown-days"
              type="text"
              value={officeLockdown.days}
              onChange={(e) =>
                setOfficeLockdown((prev) => ({
                  ...prev,
                  days: e.target.value,
                }))
              }
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="office-lockdown-intensities">
              Office Lockdown Intensities
            </Label>
            <Input
              id="office-lockdown-intensities"
              type="text"
              value={officeLockdown.intensities}
              onChange={(e) =>
                setOfficeLockdown((prev) => ({
                  ...prev,
                  intensities: e.target.value,
                }))
              }
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="office-lockdown-durations">
              Office Lockdown Durations
            </Label>
            <Input
              id="office-lockdown-durations"
              type="text"
              value={officeLockdown.durations}
              onChange={(e) =>
                setOfficeLockdown((prev) => ({
                  ...prev,
                  durations: e.target.value,
                }))
              }
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Chart Component
const SimulationChart = ({
  data,
  viewType,
  config,
}: {
  data: any;
  viewType: string;
  config: any;
}) => {
  if (!data || data.length === 0) {
    return <div className="text-center p-8">No data to display</div>;
  }

  // Process data based on viewType
  const processedData = data.map((point: any, index: number) => {
    // point structure: [S_students, I_students, R_students, S_adults, I_adults, R_adults]
    const [s_students, i_students, r_students, s_adults, i_adults, r_adults] =
      point;

    if (viewType === "students") {
      return {
        day: index,
        susceptible: s_students,
        infected: i_students,
        recovered: r_students,
      };
    } else if (viewType === "adults") {
      return {
        day: index,
        susceptible: s_adults,
        infected: i_adults,
        recovered: r_adults,
      };
    } else {
      // Combined view
      return {
        day: index,
        susceptible: s_students * config[0] + s_adults * config[1],
        infected: i_students * config[0] + i_adults * config[1],
        recovered: r_students * config[0] + r_adults * config[1],
      };
    }
  });

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={processedData}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="day"
          label={{ value: "Day", position: "insideBottomRight", offset: -5 }}
        />
        <YAxis
          domain={[0, 1]}
          label={{ value: "Population", angle: -90, position: "insideLeft" }}
        />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="susceptible"
          stroke="#8884d8"
          name="Susceptible"
          strokeWidth={2}
        />
        <Line
          type="monotone"
          dataKey="infected"
          stroke="#ff0000"
          name="Infected"
          strokeWidth={2}
        />
        <Line
          type="monotone"
          dataKey="recovered"
          stroke="#82ca9d"
          name="Recovered"
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default function Home() {
  const [initialPopulation, setInitialPopulation] = useState({
    adults: [0.99, 0.01, 0],
    students: [0.99, 0.01, 0],
  });

  const [stateConfig, setStateConfig] = useState([
    0.2, 0.8, 6, 1500, 300, 0.35, 0.14,
  ]);

  const [schoolLockdown, setSchoolLockdown] = useState({
    days: "[]",
    intensities: "[]",
    durations: "[]",
  });

  const [officeLockdown, setOfficeLockdown] = useState({
    days: "[]",
    intensities: "[]",
    durations: "[]",
  });

  const [data, setData] = useState([]);
  const [viewType, setViewType] = useState("both");
  const [loading, setLoading] = useState(false);

  const handleRunSimulation = async () => {
    setData([]);
    try {
      setLoading(true);
      const response = await axios.post("http://localhost:8000/run", {
        config: {
          sir: [...initialPopulation.students, ...initialPopulation.adults],
          static: [...stateConfig],
          school_lockdown_days: JSON.parse(schoolLockdown.days),
          school_lockdown_intensities: JSON.parse(schoolLockdown.intensities),
          school_lockdown_durations: JSON.parse(schoolLockdown.durations),
          office_lockdown_days: JSON.parse(officeLockdown.days),
          office_lockdown_intensities: JSON.parse(officeLockdown.intensities),
          office_lockdown_durations: JSON.parse(officeLockdown.durations),
        },
        model_name: "xgboost",
      });

      setData(
        response.data.result.map((x: any) =>
          x.map((y: any) => Math.min(1, Math.max(0, y)))
        )
      );
    } catch (error) {
      console.error("Error running simulation:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 max-h-[100vh]">
      <SimulationInput
        initialPopulation={initialPopulation}
        setInitialPopulation={setInitialPopulation}
        stateConfig={stateConfig}
        setStateConfig={setStateConfig}
        schoolLockdown={schoolLockdown}
        setSchoolLockdown={setSchoolLockdown}
        officeLockdown={officeLockdown}
        setOfficeLockdown={setOfficeLockdown}
      />
      <div className="p-4 space-y-4 overflow-y-auto">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Simulation Results</CardTitle>
            <Button onClick={handleRunSimulation} disabled={loading}>
              {loading ? "Running..." : "Run Simulation"}
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-center">
              <ToggleGroup
                type="single"
                value={viewType}
                onValueChange={(value) => value && setViewType(value)}
              >
                <ToggleGroupItem value="students">
                  Students Only
                </ToggleGroupItem>
                <ToggleGroupItem value="adults">Adults Only</ToggleGroupItem>
                <ToggleGroupItem value="both">Combined</ToggleGroupItem>
              </ToggleGroup>
            </div>

            <SimulationChart
              data={data}
              viewType={viewType}
              config={[stateConfig[0], stateConfig[1]]}
            />

            {data.length > 0 && (
              <div className="mt-4">
                <details>
                  <summary className="cursor-pointer font-medium">
                    Raw Data
                  </summary>
                  <div className="mt-2 max-h-60 overflow-y-auto">
                    <pre className="text-xs">
                      {JSON.stringify(data, null, 2)}
                    </pre>
                  </div>
                </details>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
