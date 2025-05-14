'use client';

import React from 'react';
import { usePathname, useRouter } from 'next/navigation';
import Link from 'next/link';

const Navigation: React.FC = () => {
  const pathname = usePathname();
  const router = useRouter();

  const isActive = (path: string) => {
    return pathname === path;
  };

  const navItems = [
    { path: '/', label: 'Home', icon: '🏠' },
    { path: '/disease-detection', label: 'Disease Detection', icon: '🍃' },
    { path: '/water-irrigation', label: 'Irrigation', icon: '💧' },
    { path: '/crop-recommendation', label: 'Crop Recommendation', icon: '🌱' },
    { path: '/yield-prediction', label: 'Yield Prediction', icon: '📊' },
  ];

  return (
    <>
      {/* Top navigation bar */}
      <header className="top-navbar">
        <div className="logo" onClick={() => router.push('/')}>
          <span className="logo-text">AgraNova</span>
        </div>
        <div className="nav-actions">
          <div className="user-profile">
            <img src="/images/logo.svg" alt="AgraNova Logo" className="avatar-logo" width={32} height={32} />
          </div>
        </div>
      </header>

      {/* Side navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h3>Smart Agriculture</h3>
        </div>
        <nav className="sidebar-nav">
          <ul className="nav-list">
            {navItems.map((item) => (
              <li key={item.path} className={isActive(item.path) ? 'active' : ''}>
                <Link href={item.path} className="nav-item">
                  <span className="nav-icon">{item.icon}</span>
                  <span className="nav-label">{item.label}</span>
                  {isActive(item.path) && <span className="active-indicator"></span>}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="sidebar-footer">
          <p>AgraNova - AI Olympiades</p>
          <p className="version">v1.0.0</p>
        </div>
      </aside>
    </>
  );
};

export default Navigation;
