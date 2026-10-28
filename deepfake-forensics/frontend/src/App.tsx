import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { EmptyState, Spinner } from './components/ui'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Analyze = lazy(() => import('./pages/Analyze'))
const Forensics = lazy(() => import('./pages/Forensics'))
const Robustness = lazy(() => import('./pages/Robustness'))
const Evaluation = lazy(() => import('./pages/Evaluation'))
const History = lazy(() => import('./pages/History'))
const Settings = lazy(() => import('./pages/Settings'))

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<Spinner />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/forensics" element={<Forensics />} />
          <Route path="/forensics/:id" element={<Forensics />} />
          <Route path="/robustness" element={<Robustness />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/history" element={<History />} />
          <Route path="/history/:id" element={<History />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<EmptyState title="Page not found" />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}
