import React from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';

function Results({ data }) {
  const { students, total_students, attentive_percentage, inattentive_percentage } = data;

  // Data for pie chart
  const chartData = [
    { name: 'Attentive', value: attentive_percentage },
    { name: 'Inattentive', value: inattentive_percentage }
  ];

  const COLORS = ['#00C49F', '#FF4444'];

  return (
    <div className="results-container">
      <h2>Analysis Results</h2>
      <p>Total Students: {total_students}</p>

      <h3>Student Breakdown</h3>
      <div className="student-grid">
        {students.map(student => (
          <div key={student.student_id} className={`student-card ${student.status}`}>
            <p>Student ID: {student.student_id}</p>
            <img src={`http://localhost:5000${student.snapshot}`} alt={`Student ${student.student_id}`} />
            <p>Score: {student.score}%</p>
            <p>Status: {student.status}</p>
          </div>
        ))}
      </div>

      <h3>Attentiveness Distribution</h3>
      <div className="pie-chart">
        <PieChart width={400} height={300}>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            label
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      </div>
    </div>
  );
}

export default Results;
